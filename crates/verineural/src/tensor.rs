use crate::{Domain, NeuralError, Operation};

/// Validated, immutable tensor dimensions. An empty dimension list is a scalar.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Shape(Vec<usize>);

impl Shape {
    pub fn new(dims: impl Into<Vec<usize>>) -> Result<Self, NeuralError> {
        let dims = dims.into();
        if dims.contains(&0) {
            return Err(NeuralError::ZeroDimension);
        }
        dims.iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or(NeuralError::ElementCountOverflow)?;
        Ok(Self(dims))
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn element_count(&self) -> usize {
        // Checked on construction and immutable afterwards.
        self.0.iter().product()
    }
}

/// Shape information for a tensor. Custom implementations are revalidated.
pub trait TensorShape: Clone {
    fn dims(&self) -> &[usize];
}

impl TensorShape for Shape {
    fn dims(&self) -> &[usize] {
        self.dims()
    }
}

/// Owned host tensor. Shape and number of elements are checked at construction.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor<T, S: TensorShape = Shape> {
    shape: S,
    values: Vec<T>,
}

impl<T, S: TensorShape> Tensor<T, S> {
    pub fn new(shape: S, values: Vec<T>) -> Result<Self, NeuralError> {
        let checked = Shape::new(shape.dims().to_vec())?;
        if checked.element_count() != values.len() {
            return Err(NeuralError::DataLengthMismatch {
                expected: checked.element_count(),
                actual: values.len(),
            });
        }
        Ok(Self { shape, values })
    }

    pub fn shape(&self) -> &S {
        &self.shape
    }

    pub fn values(&self) -> &[T] {
        &self.values
    }

    pub fn into_values(self) -> Vec<T> {
        self.values
    }
}

impl Tensor<f32> {
    /// Independently validate concrete input values against a graph assumption.
    pub fn validate_domain(&self, domain: Domain) -> Result<(), NeuralError> {
        for (index, &v) in self.values.iter().enumerate() {
            let valid = match domain {
                Domain::Any => true,
                Domain::Finite => v.is_finite(),
                Domain::NonNegativeFinite => v.is_finite() && v >= 0.0,
            };
            if !valid {
                return Err(NeuralError::ValueOutsideDomain {
                    index,
                    required: domain,
                });
            }
        }
        Ok(())
    }

    /// Pure host reference calculation. Not a certified FP32 kernel.
    pub fn add(&self, rhs: &Self) -> Result<Self, NeuralError> {
        let shape = Operation::Add.infer_shape(&[self.shape.clone(), rhs.shape.clone()])?;
        let values = self
            .values
            .iter()
            .zip(&rhs.values)
            .map(|(&a, &b)| a + b)
            .collect();
        Self::new(shape, values)
    }

    /// Pure elementwise host reference calculation, without broadcasting.
    pub fn mul(&self, rhs: &Self) -> Result<Self, NeuralError> {
        let shape = Operation::Mul.infer_shape(&[self.shape.clone(), rhs.shape.clone()])?;
        let values = self
            .values
            .iter()
            .zip(&rhs.values)
            .map(|(&a, &b)| a * b)
            .collect();
        Self::new(shape, values)
    }

    /// Add one rank-1 bias vector to every matrix row.
    pub fn bias_add(&self, rhs: &Self) -> Result<Self, NeuralError> {
        let shape = Operation::BiasAdd.infer_shape(&[self.shape.clone(), rhs.shape.clone()])?;
        let columns = shape.dims()[1];
        let values = self
            .values
            .iter()
            .enumerate()
            .map(|(i, &x)| x + rhs.values[i % columns])
            .collect();
        Self::new(shape, values)
    }

    /// Matrix multiplication of two rank-2 tensors. Accumulates left to right.
    pub fn matmul(&self, rhs: &Self) -> Result<Self, NeuralError> {
        let shape = Operation::MatMul.infer_shape(&[self.shape.clone(), rhs.shape.clone()])?;
        let [m, k] = self.shape.dims() else {
            unreachable!("shape checked above")
        };
        let n = rhs.shape.dims()[1];
        let mut values = vec![0.0f32; shape.element_count()];
        for row in 0..*m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for inner in 0..*k {
                    sum += self.values[row * *k + inner] * rhs.values[inner * n + col];
                }
                values[row * n + col] = sum;
            }
        }
        Self::new(shape, values)
    }

    /// Canonical 0/1 positive mask: NaN and signed zero both take false.
    /// This host operation is a reference, not an IEEE/GPU proof.
    pub fn positive_mask(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            values: self
                .values
                .iter()
                .map(|&v| if v > 0.0 { 1.0 } else { 0.0 })
                .collect(),
        }
    }

    /// Select the original binary32 payload from one branch, without arithmetic,
    /// using a strict positive comparison. The result has the branches' shape.
    pub fn select_positive(&self, yes: &Self, no: &Self) -> Result<Self, NeuralError> {
        let shape = Operation::SelectPositive.infer_shape(&[
            self.shape.clone(),
            yes.shape.clone(),
            no.shape.clone(),
        ])?;
        let values = self
            .values
            .iter()
            .zip(&yes.values)
            .zip(&no.values)
            .map(
                |((&mask, &on_true), &on_false)| {
                    if mask > 0.0 { on_true } else { on_false }
                },
            )
            .collect();
        Self::new(shape, values)
    }

    /// Elementwise max(x, 0), preserving NaN instead of silently hiding it.
    pub fn relu(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            values: self
                .values
                .iter()
                .map(|&v| if v.is_nan() || v > 0.0 { v } else { 0.0 })
                .collect(),
        }
    }

    /// Reject negative and non-finite arguments, as specified by the IR domain.
    pub fn sqrt(&self) -> Result<Self, NeuralError> {
        self.validate_domain(Domain::NonNegativeFinite)?;
        Self::new(
            self.shape.clone(),
            self.values.iter().map(|v| v.sqrt()).collect(),
        )
    }

    /// Same values, new dimensions with the same element count.
    pub fn reshape(self, shape: Shape) -> Result<Self, NeuralError> {
        Operation::Reshape(shape.clone()).infer_shape(std::slice::from_ref(&self.shape))?;
        Self::new(shape, self.values)
    }
}
