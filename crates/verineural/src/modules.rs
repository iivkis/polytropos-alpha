//! Stage 6 neural building blocks expressed entirely through Semantic Neural IR.
//!
//! These modules have checked *shape* contracts; f32 overflow and backend
//! lowering remain outside the current Lean proof boundary.

use crate::{MathContract, NeuralError, NeuralGraph, NeuralModule, NodeId, PortContract, Shape};

/// Dense layer Y = X * W + bias, with row-wise bias broadcast.
pub struct Dense {
    contract: MathContract,
}

impl Dense {
    /// X is [batch, inputs], W is [inputs, outputs], bias is [outputs].
    pub fn new(batch: usize, inputs: usize, outputs: usize) -> Result<Self, NeuralError> {
        let x = Shape::new(vec![batch, inputs])?;
        let w = Shape::new(vec![inputs, outputs])?;
        let bias = Shape::new(vec![outputs])?;
        let y = Shape::new(vec![batch, outputs])?;
        Ok(Self {
            contract: MathContract::new(
                vec![
                    PortContract::any(x),
                    PortContract::any(w),
                    PortContract::any(bias),
                ],
                PortContract::any(y),
            ),
        })
    }
}

impl NeuralModule for Dense {
    fn name(&self) -> &'static str {
        "dense"
    }

    fn contract(&self) -> &MathContract {
        &self.contract
    }

    fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        let product = graph.matmul(inputs[0], inputs[1])?;
        graph.bias_add(product, inputs[2])
    }
}

/// Elementwise residual connection Y = X + F(X), with no broadcasting.
pub struct ResidualAdd {
    contract: MathContract,
}

impl ResidualAdd {
    pub fn new(shape: Shape) -> Self {
        Self {
            contract: MathContract::new(
                vec![
                    PortContract::any(shape.clone()),
                    PortContract::any(shape.clone()),
                ],
                PortContract::any(shape),
            ),
        }
    }
}

impl NeuralModule for ResidualAdd {
    fn name(&self) -> &'static str {
        "residual_add"
    }

    fn contract(&self) -> &MathContract {
        &self.contract
    }

    fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        graph.add(inputs[0], inputs[1])
    }
}

/// Residual two-layer FFN:
/// X + (ReLU(X W1 + b1) W2 + b2)
///
/// The five inputs are X, W1, b1, W2 and b2 in that order.
/// This is a forward-only composable primitive, not a Transformer or training
/// implementation. All weights are explicit graph inputs, not hidden constants.
pub struct ResidualFfn {
    contract: MathContract,
}

impl ResidualFfn {
    pub fn new(batch: usize, features: usize, hidden: usize) -> Result<Self, NeuralError> {
        let x = Shape::new(vec![batch, features])?;
        let w1 = Shape::new(vec![features, hidden])?;
        let b1 = Shape::new(vec![hidden])?;
        let w2 = Shape::new(vec![hidden, features])?;
        let b2 = Shape::new(vec![features])?;
        Ok(Self {
            contract: MathContract::new(
                vec![
                    PortContract::any(x.clone()),
                    PortContract::any(w1),
                    PortContract::any(b1),
                    PortContract::any(w2),
                    PortContract::any(b2),
                ],
                PortContract::any(x),
            ),
        })
    }
}

impl NeuralModule for ResidualFfn {
    fn name(&self) -> &'static str {
        "residual_ffn"
    }

    fn contract(&self) -> &MathContract {
        &self.contract
    }

    fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        let first = graph.matmul(inputs[0], inputs[1])?;
        let first = graph.bias_add(first, inputs[2])?;
        let activated = graph.relu(first)?;
        let projected = graph.matmul(activated, inputs[3])?;
        let projected = graph.bias_add(projected, inputs[4])?;
        graph.add(inputs[0], projected)
    }
}
