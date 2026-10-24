//! Real CUDA Graph subset H2D and D2H batch parity.
//! One explicit non-blocking stream, queued pinned transfers and one barrier.
use verineural::{Domain,GpuEngine,GpuError,NeuralGraph,Shape,Tensor};

#[test]
#[ignore="requires NVIDIA GPU and LLVM NVPTX; checks batch parity and fail-before-DMA validation"]
fn batched_subset_transfers_match_single_updates_and_reads() {
    let mut g=NeuralGraph::default();
    let shape=Shape::new(vec![4]).unwrap();
    let x=g.input(shape.clone(),Domain::Finite);
    let y=g.input(shape,Domain::Finite);
    let sum=g.add(x,y).unwrap();
    let product=g.mul(sum,y).unwrap();
    let arch=std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_|"sm_52".to_string());
    let engine=GpuEngine::new("llc",arch);
    let program=engine.compile_outputs(&g,&[sum,product]).unwrap();
    let initial=[
        (x,Tensor::from_slice(vec![4],&[1.,-2.,0.,8.]).unwrap()),
        (y,Tensor::from_slice(vec![4],&[3.,4.,-0.5,2.]).unwrap()),
    ];
    let dir=std::env::temp_dir().join(format!("vn_batch_transfers_{}",std::process::id()));
    let mut gpu=engine.prepare(&program,&initial,&dir).unwrap();
    gpu.replay().unwrap();

    let old=gpu.read_outputs().unwrap();
    let reversed=gpu.read_outputs_selected(&[product,sum]).unwrap();
    assert_eq!(old[0].0,sum);
    assert_eq!(old[1].0,product);
    assert_eq!(reversed[0].0,product);
    assert_eq!(reversed[1].0,sum);
    for (left,right) in old.iter().zip(reversed.iter().rev()) {
        assert_eq!(left.0,right.0);
        for (a,b) in left.1.values().iter().zip(right.1.values()) {
            assert_eq!(a.to_bits(),b.to_bits());
        }
    }

    assert!(matches!(
        gpu.read_outputs_selected(&[sum,sum]),
        Err(GpuError::InvalidInput{reason:"duplicate selected read output",..})
    ));
    assert!(matches!(
        gpu.read_outputs_selected(&[x]),
        Err(GpuError::InvalidInput{reason:"requested output is not selected",..})
    ));

    let new_x=Tensor::from_slice(vec![4],&[-2.,1.,4.,0.]).unwrap();
    let new_y=Tensor::from_slice(vec![4],&[3.,-4.,0.25,2.]).unwrap();
    let invalid=Tensor::from_slice(vec![2],&[5.,6.]).unwrap();
    assert!(matches!(
        gpu.update_input_subset(&[(x,&new_x),(x,&new_y)]),
        Err(GpuError::InvalidInput{reason:"duplicate partial input update",..})
    ));
    assert!(gpu.update_input_subset(&[(x,&new_x),(y,&invalid)]).is_err());
    // A rejected batch must not invalidate the prior outputs.
    let unmodified=gpu.read_outputs_selected(&[sum,product]).unwrap();
    for (a,b) in unmodified.iter().zip(&old) {
        for (v,w) in a.1.values().iter().zip(b.1.values()) {
            assert_eq!(v.to_bits(),w.to_bits());
        }
    }
    gpu.update_input_subset(&[(x,&new_x),(y,&new_y)]).unwrap();
    gpu.replay().unwrap();
    let actual=gpu.read_outputs_selected(&[sum,product]).unwrap();
    for i in 0..4 {
        let sum_expected=new_x.values()[i]+new_y.values()[i];
        let product_expected=sum_expected*new_y.values()[i];
        assert_eq!(actual[0].1.values()[i].to_bits(),sum_expected.to_bits());
        assert_eq!(actual[1].1.values()[i].to_bits(),product_expected.to_bits());
    }
    // Existing pointwise single-update API still yields the same results.
    gpu.update_input(x,&new_x).unwrap();
    gpu.update_input(y,&new_y).unwrap();
    gpu.replay().unwrap();
    let individual=gpu.read_outputs().unwrap();
    for (a,b) in actual.iter().zip(&individual) {
        for (v,w) in a.1.values().iter().zip(b.1.values()) {
            assert_eq!(v.to_bits(),w.to_bits());
        }
    }
}
