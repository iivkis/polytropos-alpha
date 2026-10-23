use crate::{ArgMaxTieBreak, Domain, NeuralError, Operation};

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

fn unravel_index(mut index: usize, dims: &[usize]) -> Vec<usize> {
    let mut coords = vec![0; dims.len()];
    for axis in (0..dims.len()).rev() {
        coords[axis] = index % dims[axis];
        index /= dims[axis];
    }
    coords
}

fn ravel_index(coords: &[usize], dims: &[usize]) -> usize {
    coords
        .iter()
        .zip(dims)
        .fold(0usize, |flat, (&coord, &dim)| flat * dim + coord)
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

impl<T> Tensor<T> {
    /// Construct a tensor without creating a Shape separately.
    pub fn from_vec(dims: impl Into<Vec<usize>>, values: Vec<T>) -> Result<Self, NeuralError> {
        Self::new(Shape::new(dims)?, values)
    }

    /// Copy borrowed values into a shape-validated tensor.
    pub fn from_slice(dims: impl Into<Vec<usize>>, values: &[T]) -> Result<Self, NeuralError>
    where
        T: Clone,
    {
        Self::from_vec(dims, values.to_vec())
    }

    /// Create a tensor filled with T::default(), zero for numeric types.
    pub fn zeros(dims: impl Into<Vec<usize>>) -> Result<Self, NeuralError>
    where
        T: Default + Clone,
    {
        let shape = Shape::new(dims)?;
        Self::new(shape.clone(), vec![T::default(); shape.element_count()])
    }
}

impl<T: Clone> Tensor<T> {
    /// Reorder axes in row-major order. `axes[output_axis] = input_axis`.
    pub fn permute(&self, axes: &[usize]) -> Result<Self, NeuralError> {
        let operation = Operation::Permute {
            axes: axes.to_vec(),
        };
        let shape = operation.infer_shape(std::slice::from_ref(&self.shape))?;
        let mut values = Vec::with_capacity(self.values.len());
        for output_index in 0..shape.element_count() {
            let output_coords = unravel_index(output_index, shape.dims());
            let mut input_coords = vec![0; self.shape.rank()];
            for (output_axis, &input_axis) in axes.iter().enumerate() {
                input_coords[input_axis] = output_coords[output_axis];
            }
            values.push(self.values[ravel_index(&input_coords, self.shape.dims())].clone());
        }
        Self::new(shape, values)
    }

    /// Explicit right-aligned broadcast; singleton input axes repeat.
    pub fn broadcast_to(&self, output: Shape) -> Result<Self, NeuralError> {
        let op = Operation::BroadcastTo(output);
        let shape = op.infer_shape(std::slice::from_ref(&self.shape))?;
        let offset = shape.rank() - self.shape.rank();
        let mut values = Vec::with_capacity(shape.element_count());
        for index in 0..shape.element_count() {
            let out = unravel_index(index, shape.dims());
            let coords = self
                .shape
                .dims()
                .iter()
                .enumerate()
                .map(
                    |(axis, &width)| {
                        if width == 1 { 0 } else { out[offset + axis] }
                    },
                )
                .collect::<Vec<_>>();
            values.push(self.values[ravel_index(&coords, self.shape.dims())].clone());
        }
        Self::new(shape, values)
    }

    /// Reverse one row-major axis without changing shape.
    pub fn reverse_axis(&self, axis: usize) -> Result<Self, NeuralError> {
        let operation = Operation::ReverseAxis { axis };
        let shape = operation.infer_shape(std::slice::from_ref(&self.shape))?;
        let mut values = Vec::with_capacity(self.values.len());
        for output_index in 0..shape.element_count() {
            let mut input_coords = unravel_index(output_index, shape.dims());
            input_coords[axis] = shape.dims()[axis] - 1 - input_coords[axis];
            values.push(self.values[ravel_index(&input_coords, self.shape.dims())].clone());
        }
        Self::new(shape, values)
    }

    /// Concatenate two tensors along one axis.
    pub fn concat(&self, rhs: &Self, axis: usize) -> Result<Self, NeuralError> {
        let operation = Operation::Concat { axis };
        let shape = operation.infer_shape(&[self.shape.clone(), rhs.shape.clone()])?;
        let left_width = self.shape.dims()[axis];
        let mut values = Vec::with_capacity(shape.element_count());
        for output_index in 0..shape.element_count() {
            let output_coords = unravel_index(output_index, shape.dims());
            if output_coords[axis] < left_width {
                values.push(self.values[ravel_index(&output_coords, self.shape.dims())].clone());
            } else {
                let mut right_coords = output_coords;
                right_coords[axis] -= left_width;
                values.push(rhs.values[ravel_index(&right_coords, rhs.shape.dims())].clone());
            }
        }
        Self::new(shape, values)
    }

    /// Select one lane from an axis using a one-element host index tensor.
    pub fn take(&self, index: &Tensor<usize>, axis: usize) -> Result<Self, NeuralError> {
        let operation = Operation::Take { axis };
        let shape = operation.infer_shape(&[self.shape.clone(), index.shape.clone()])?;
        let selected = index.values[0];
        let bound = self.shape.dims()[axis];
        if selected >= bound {
            return Err(NeuralError::IndexOutOfBounds {
                axis,
                index: selected,
                bound,
            });
        }
        let mut values = Vec::with_capacity(shape.element_count());
        for output_index in 0..shape.element_count() {
            let output_coords = unravel_index(output_index, shape.dims());
            let mut input_coords = Vec::with_capacity(self.shape.rank());
            let mut output_axis = 0;
            for input_axis in 0..self.shape.rank() {
                if input_axis == axis {
                    input_coords.push(selected);
                } else {
                    input_coords.push(output_coords[output_axis]);
                    output_axis += 1;
                }
            }
            values.push(self.values[ravel_index(&input_coords, self.shape.dims())].clone());
        }
        Self::new(shape, values)
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

    /// Elementwise subtraction, without implicit broadcasting.
    pub fn sub(&self, rhs: &Self) -> Result<Self, NeuralError> {
        let shape = Operation::Add.infer_shape(&[self.shape.clone(), rhs.shape.clone()])?;
        Self::new(
            shape,
            self.values
                .iter()
                .zip(&rhs.values)
                .map(|(&a, &b)| a - b)
                .collect(),
        )
    }

    /// Elementwise division (including IEEE NaN/inf on zero denominator).
    pub fn div(&self, rhs: &Self) -> Result<Self, NeuralError> {
        let shape = Operation::Mul.infer_shape(&[self.shape.clone(), rhs.shape.clone()])?;
        Self::new(
            shape,
            self.values
                .iter()
                .zip(&rhs.values)
                .map(|(&a, &b)| a / b)
                .collect(),
        )
    }

    /// Elementwise exponential. Host f32 reference, not certified.
    pub fn exp(&self) -> Result<Self, NeuralError> {
        Self::new(
            self.shape.clone(),
            self.values.iter().map(|x| x.exp()).collect(),
        )
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

    /// Deterministic row-major argmax. Equal maxima keep the first lane. NaN
    /// never replaces a numeric best; an initial NaN is replaced by the first
    /// numeric lane. If every lane is NaN, the first lane wins.
    pub fn argmax(
        &self,
        axis: usize,
        tie_break: ArgMaxTieBreak,
    ) -> Result<Tensor<usize>, NeuralError> {
        let operation = Operation::ArgMax { axis, tie_break };
        let shape = operation.infer_shape(std::slice::from_ref(&self.shape))?;
        let width = self.shape.dims()[axis];
        let inner = self.shape.dims()[axis + 1..].iter().product::<usize>();
        let mut values = Vec::with_capacity(shape.element_count());
        for output_index in 0..shape.element_count() {
            let outer = output_index / inner;
            let suffix = output_index % inner;
            let mut best = 0usize;
            let mut best_value = self.values[(outer * width) * inner + suffix];
            for lane in 1..width {
                let candidate = self.values[(outer * width + lane) * inner + suffix];
                match tie_break {
                    ArgMaxTieBreak::First => {
                        if !candidate.is_nan() && (best_value.is_nan() || candidate > best_value) {
                            best = lane;
                            best_value = candidate;
                        }
                    }
                }
            }
            values.push(best);
        }
        Tensor::new(shape, values)
    }

    /// Row-major sum over any axis, with an explicit left-to-right f32 fold.
    /// This is a host reference, NOT a proof of its rounding or GPU behavior.
    pub fn reduce_sum(&self, axis: usize) -> Result<Self, NeuralError> {
        let output_shape =
            Operation::ReduceSum { axis }.infer_shape(std::slice::from_ref(&self.shape))?;
        let width = self.shape.dims()[axis];
        let inner = self.shape.dims()[axis + 1..].iter().product::<usize>();
        let mut values = Vec::with_capacity(output_shape.element_count());
        for index in 0..output_shape.element_count() {
            let outer = index / inner;
            let suffix = index % inner;
            let mut sum = 0.0f32;
            for lane in 0..width {
                sum += self.values[(outer * width + lane) * inner + suffix];
            }
            values.push(sum);
        }
        Self::new(output_shape, values)
    }

    /// Deterministic maximum, preserving NaN and taking the first maximum
    /// on ties. f32 semantics are reference-only until a backend is audited.
    pub fn reduce_max(&self, axis: usize) -> Result<Self, NeuralError> {
        let output_shape =
            Operation::ReduceMax { axis }.infer_shape(std::slice::from_ref(&self.shape))?;
        let width = self.shape.dims()[axis];
        let inner = self.shape.dims()[axis + 1..].iter().product::<usize>();
        let mut values = Vec::with_capacity(output_shape.element_count());
        for index in 0..output_shape.element_count() {
            let outer = index / inner;
            let suffix = index % inner;
            let mut best = self.values[(outer * width) * inner + suffix];
            for lane in 1..width {
                let candidate = self.values[(outer * width + lane) * inner + suffix];
                if candidate.is_nan() || (!best.is_nan() && candidate > best) {
                    best = candidate;
                }
            }
            values.push(best);
        }
        Self::new(output_shape, values)
    }

    /// Same values, new dimensions with the same element count.
    pub fn reshape(self, shape: Shape) -> Result<Self, NeuralError> {
        Operation::Reshape(shape.clone()).infer_shape(std::slice::from_ref(&self.shape))?;
        Self::new(shape, self.values)
    }
}
