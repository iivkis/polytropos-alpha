//! P1/P10 explicit contiguous layout and zero-copy alias semantics.

use verineural::{Domain, GpuEngine, NeuralGraph, NodeStorage, Shape};

#[test]
fn shape_exposes_canonical_row_major_strides() {
    assert_eq!(
        Shape::new([2, 3, 4]).unwrap().row_major_strides(),
        [12, 4, 1]
    );
    assert_eq!(
        Shape::new(Vec::<usize>::new()).unwrap().row_major_strides(),
        []
    );
}

#[test]
fn reshape_alias_chain_resolves_to_one_storage_root() {
    let mut graph = NeuralGraph::default();
    let input = graph.input(Shape::new([2, 3]).unwrap(), Domain::Finite);
    let first = graph.reshape(input, Shape::new([3, 2]).unwrap()).unwrap();
    let second = graph.reshape(first, Shape::new([6]).unwrap()).unwrap();

    assert_eq!(
        graph.storage(input).unwrap(),
        NodeStorage::OwnedContiguous {
            strides: vec![3, 1]
        }
    );
    assert_eq!(
        graph.storage(first).unwrap(),
        NodeStorage::Alias {
            source: input,
            strides: vec![2, 1]
        }
    );
    assert_eq!(
        graph.storage(second).unwrap(),
        NodeStorage::Alias {
            source: first,
            strides: vec![1]
        }
    );
    assert_eq!(graph.storage_root(second).unwrap(), input);
    assert!(graph.aliases(input, second).unwrap());
}

#[test]
fn data_reordering_ops_own_contiguous_storage_instead_of_pretending_to_be_views() {
    let mut graph = NeuralGraph::default();
    let input = graph.input(Shape::new([2, 3]).unwrap(), Domain::Finite);
    let transposed = graph.permute(input, &[1, 0]).unwrap();
    let sliced = graph.slice_axis(input, 1, 1, 2).unwrap();

    assert_eq!(
        graph.storage(transposed).unwrap(),
        NodeStorage::OwnedContiguous {
            strides: vec![2, 1]
        }
    );
    assert_eq!(
        graph.storage(sliced).unwrap(),
        NodeStorage::OwnedContiguous {
            strides: vec![2, 1]
        }
    );
    assert!(!graph.aliases(input, transposed).unwrap());
    assert!(!graph.aliases(input, sliced).unwrap());
}

#[test]
fn canonical_ir_v4_binds_layout_and_alias_identity() {
    let mut graph = NeuralGraph::default();
    let input = graph.input(Shape::new([2, 3]).unwrap(), Domain::Finite);
    let reshaped = graph.reshape(input, Shape::new([3, 2]).unwrap()).unwrap();
    let output = graph.relu(reshaped).unwrap();

    let ir = graph.canonical_ir();
    assert!(ir.starts_with("verineural-semantic-ir-v4\n"));
    assert!(ir.contains("storage=owned_contiguous:strides=[3, 1]"));
    assert!(ir.contains("storage=alias_node_0:strides=[2, 1]"));

    let program = GpuEngine::default().compile(&graph, output).unwrap();
    let capability = program.capability_plan().unwrap();
    assert_eq!(capability.kernels.len(), 1);
    assert_eq!(capability.kernels[0].operation, "relu");
    assert_eq!(capability.kernels[0].reads, vec![(reshaped.index(), 6)]);
}
