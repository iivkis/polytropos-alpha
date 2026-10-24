//! Checked opt-in Mul -> Add GPU fusion, including reverse-mode and
//! selected-output / fan-out protection. Explicit mul.rn then add.rn, NOT FMA.
use verineural::{
    Domain, GpuEngine, GpuLaunchConfig, NeuralGraph, Shape, Tensor,
    TranslationKind, TranslationPlan, differentiate,
};

fn graph(mul_on_right: bool) -> (NeuralGraph, [verineural::NodeId; 3], verineural::NodeId, verineural::NodeId) {
    let mut graph = NeuralGraph::default();
    let shape=Shape::new(vec![1,8]).unwrap();
    let a=graph.input(shape.clone(),Domain::Finite);
    let b=graph.input(shape.clone(),Domain::Finite);
    let c=graph.input(shape,Domain::Finite);
    let mul=graph.mul(a,b).unwrap();
    let root=if mul_on_right { graph.add(c,mul).unwrap() } else {graph.add(mul,c).unwrap()};
    (graph,[a,b,c],mul,root)
}

#[test]
fn fusion_is_explicit_versioned_and_keeps_selected_or_shared_values() {
    let config=GpuLaunchConfig::default().with_pointwise_fusion();
    assert!(config.pointwise_fusion());
    for rhs in [false,true] {
        let (g,_,mul,root)=graph(rhs);
        let plain=GpuEngine::default().compile(&g,root).unwrap();
        let optimized=GpuEngine::default().with_launch_config(config).compile(&g,root).unwrap();
        assert_eq!(plain.kernel_count().unwrap(),2);
        assert_eq!(optimized.kernel_count().unwrap(),1);
        assert_eq!(optimized.removed_kernel_count(),1);
        assert_eq!(optimized.fused_mul_adds()[&root.index()],(mul.index(),usize::from(rhs)));
        let plan=optimized.translation_plan().unwrap();
        assert_eq!(plan.kernels.len(),1);
        assert_eq!(plan.kernels[0].kind,TranslationKind::MulAdd{multiply_on_right:rhs});
        assert_eq!(TranslationPlan::from_text(&plan.to_text()).unwrap(),plan);
        assert!(optimized.llvm_ir().contains("mul.rn.f32"));
        assert!(optimized.llvm_ir().contains("add.rn.f32"));
        assert!(!optimized.llvm_ir().contains("fma.rn.f32"));
        assert_eq!(optimized.capability_plan().unwrap().optimizations.len(),1);

        // An independently selected producer cannot be folded away.
        let observable=GpuEngine::default().with_launch_config(config)
            .compile_outputs(&g,&[root,mul]).unwrap();
        assert!(observable.fused_mul_adds().is_empty());
        assert_eq!(observable.kernel_count().unwrap(),2);

        // Two reachable consumers likewise prevent fusion.
        let mut shared=g.clone();
        let other=shared.add(mul,root).unwrap();
        let protected=GpuEngine::default().with_launch_config(config)
            .compile(&shared,other).unwrap();
        assert!(protected.fused_mul_adds().is_empty());
    }
}

#[test]
fn opt_in_reduces_kernels_on_a_reverse_mode_graph() {
    let (g,[a,b,c],_,root)=graph(false);
    let reverse=differentiate(&g,root,&[a,b,c]).unwrap();
    let gradients=reverse.gradients().iter().map(|(_,grad)|*grad).collect::<Vec<_>>();
    let regular=GpuEngine::default().compile_outputs(reverse.graph(),&gradients).unwrap();
    let optimized=GpuEngine::default().with_launch_config(
        GpuLaunchConfig::default().with_pointwise_fusion()
    ).compile_outputs(reverse.graph(),&gradients).unwrap();
    assert!(optimized.kernel_count().unwrap() <= regular.kernel_count().unwrap());
    assert!(optimized.capability_plan().is_ok());
}

#[test]
#[ignore="requires NVIDIA GPU and LLVM NVPTX for fused/unfused parity"]
fn cuda_mul_add_preserves_operand_order_and_matches_unfused() {
    let config=GpuLaunchConfig::default().with_pointwise_fusion();
    for rhs in [false,true] {
        let (g,inputs,_,root)=graph(rhs);
        let values=[
            [0.5, -1.25, 3., -4., 0., 2.2, 5.5, -0.125],
            [-0.25, 0.75, -1., 6., -0.0, -0.5, 0.25, 7.],
            [1., -2., 3., -5., 0., 1.1, -0.5, 0.25],
        ];
        let bound=inputs.iter().zip(values).map(|(&id,v)|
            (id,Tensor::from_slice(vec![1,8],&v).unwrap())
        ).collect::<Vec<_>>();
        let path=std::env::temp_dir().join(format!("vn_mul_add_{}_{}",std::process::id(),rhs));
        let ordinary=GpuEngine::new("llc",
            std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_|"sm_52".into()));
        let fused=GpuEngine::new("llc",
            std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_|"sm_52".into()))
            .with_launch_config(config);
        let old=ordinary.compile(&g,root).unwrap();
        let new=fused.compile(&g,root).unwrap();
        let old_output=ordinary.execute(&old,&bound,&path.join("plain")).unwrap();
        let new_output=fused.execute(&new,&bound,&path.join("fused")).unwrap();
        for (i,(x,y)) in old_output.values().iter().zip(new_output.values()).enumerate(){
            assert_eq!(x.to_bits(),y.to_bits(),"lane={i} side={rhs}: {x} vs {y}");
        }
    }
}


#[test]
#[ignore="requires NVIDIA GPU and LLVM NVPTX for reverse-mode bitwise parity"]
fn cuda_reverse_mode_graph_preserves_gradients_with_mul_add_fusion() {
    let (mut g,[a,b,c],_,sum)=graph(false);
    let shape=Shape::new(vec![1,8]).unwrap();
    let u=g.input(shape,Domain::Finite);
    let root=g.mul(sum,u).unwrap();
    let reverse=differentiate(&g,root,&[u,a,b]).unwrap();
    let mut outputs=vec![root];
    outputs.extend(reverse.gradients().iter().map(|(_,id)|*id));
    let config=GpuLaunchConfig::default().with_pointwise_fusion();
    let normal=GpuEngine::new("llc",
        std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_|"sm_52".into()));
    let experimental=GpuEngine::new("llc",
        std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_|"sm_52".into()))
        .with_launch_config(config);
    let base_program=normal.compile_outputs(reverse.graph(),&outputs).unwrap();
    let fused_program=experimental.compile_outputs(reverse.graph(),&outputs).unwrap();
    assert!(fused_program.kernel_count().unwrap()<base_program.kernel_count().unwrap());

    let bound=[
        (a,[0.5,1.2,3.4,-0.4,-2.,1.,0.,7.]),
        (b,[-0.5,1.5,0.2,8.,1.,-0.25,2.,1.]),
        (c,[0.25,-2.,0.125,0.7,1.2,4.,0.,-0.5]),
        (u,[0.1,0.2,0.3,-1.5,2.,4.,-0.25,0.75]),
    ].into_iter().map(|(id,v)| (id,Tensor::from_slice(vec![1,8],&v).unwrap()))
        .collect::<Vec<_>>();
    let seed=Tensor::from_vec(vec![1,8],vec![1.;8]).unwrap();
    let inputs=reverse.bindings(&bound,seed).unwrap();
    let old=inputs.iter().filter(|(id,_)|base_program.source_inputs().contains(id))
        .cloned().collect::<Vec<_>>();
    let new=inputs.iter().filter(|(id,_)|fused_program.source_inputs().contains(id))
        .cloned().collect::<Vec<_>>();
    let dir=std::env::temp_dir().join(format!("vn_mul_add_grad_{}",std::process::id()));
    let left=normal.execute_outputs(&base_program,&old,&dir.join("baseline")).unwrap();
    let right=experimental.execute_outputs(&fused_program,&new,&dir.join("fused")).unwrap();
    for (l,r) in left.iter().zip(&right) {
        assert_eq!(l.0,r.0);
        for (i,(x,y)) in l.1.values().iter().zip(r.1.values()).enumerate() {
            assert_eq!(x.to_bits(),y.to_bits(),"gradient node={:?} lane={i}",l.0);
        }
    }
}
