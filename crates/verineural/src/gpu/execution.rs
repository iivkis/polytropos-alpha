//! Prepared CUDA Graph execution and reusable state/feedback transfers.

use super::*;

impl GpuExecution {
    /// Replace all reachable graph inputs without rebuilding the CUDA Graph.
    /// Validate the full batch before the first transfer, so rejected input
    /// metadata cannot partially replace the previous input set.
    /// Transfers finish on the owned non-blocking stream before returning.
    pub fn update_inputs(&mut self, inputs: &[(NodeId, Tensor<f32>)]) -> Result<(), GpuError> {
        self.execution_health.ensure_ready()?;
        let given = validate_inputs(&self.program, inputs)?;
        self.execution_health.invalidate_outputs();
        self.preserved_output_valid.set(false);
        for id in self.program.source_inputs() {
            self.execution_health
                .record_io(self.session.copy_to_device(
                    self.input_addresses[&id.index()],
                    given[&id.index()].values(),
                ))?;
        }
        self.execution_health
            .record_io(self.session.flush_input_copies())
    }

    /// Update an arbitrary *subset* of reachable F32 source tensors.
    /// All IDs, uniqueness, shapes and domains are checked before enqueueing
    /// the first H2D copy. Pinned host staging is kept alive until a single
    /// synchronization of the owned non-blocking stream. All other inputs
    /// retain their previous device values.
    pub fn update_input_subset(
        &mut self,
        inputs: &[(NodeId, &Tensor<f32>)],
    ) -> Result<(), GpuError> {
        self.update_input_subset_impl(inputs, false)
    }

    /// Experimental packed H2D upload, using one DMA and a cached CUDA Graph
    /// scatter. Validates the complete subset before any upload.
    pub fn update_input_subset_packed(
        &mut self,
        inputs: &[(NodeId, &Tensor<f32>)],
    ) -> Result<(), GpuError> {
        self.update_input_subset_impl(inputs, true)
    }

    fn update_input_subset_impl(
        &mut self,
        inputs: &[(NodeId, &Tensor<f32>)],
        packed: bool,
    ) -> Result<(), GpuError> {
        self.execution_health.ensure_ready()?;
        let mut seen = BTreeSet::new();
        for &(id, value) in inputs {
            if !seen.insert(id.index()) {
                return Err(GpuError::InvalidInput {
                    node: id.index(),
                    reason: "duplicate partial input update",
                });
            }
            if !self.input_addresses.contains_key(&id.index()) {
                return Err(GpuError::InvalidInput {
                    node: id.index(),
                    reason: "source is outside the compiled output slice",
                });
            }
            let node = self.program.graph.node(id)?;
            if value.shape() != &node.shape {
                return Err(GpuError::InvalidGraph(NeuralError::ShapeMismatch {
                    expected: node.shape.clone(),
                    actual: value.shape().clone(),
                }));
            }
            value.validate_domain(node.domain)?;
        }
        if inputs.is_empty() {
            return Ok(());
        }
        self.execution_health.invalidate_outputs();
        self.preserved_output_valid.set(false);
        if packed {
            let transfers = inputs
                .iter()
                .map(|&(id, value)| (self.input_addresses[&id.index()], value.values()))
                .collect::<Vec<_>>();
            self.execution_health
                .record_io(self.session.copy_f32_batch_to_device_packed(&transfers))
        } else {
            for &(id, value) in inputs {
                self.execution_health.record_io(
                    self.session
                        .copy_to_device(self.input_addresses[&id.index()], value.values()),
                )?;
            }
            self.execution_health
                .record_io(self.session.flush_input_copies())
        }
    }

    /// Rebind one named source without uploading unchanged parameters or state.
    /// Metadata and domain are checked before touching GPU memory.
    pub fn update_input(&mut self, id: NodeId, value: &Tensor<f32>) -> Result<(), GpuError> {
        self.execution_health.ensure_ready()?;
        let node = self.program.graph.node(id)?;
        if !self.input_addresses.contains_key(&id.index()) {
            return Err(GpuError::InvalidInput {
                node: id.index(),
                reason: "source is outside the compiled output slice",
            });
        }
        if value.shape() != &node.shape {
            return Err(GpuError::InvalidGraph(NeuralError::ShapeMismatch {
                expected: node.shape.clone(),
                actual: value.shape().clone(),
            }));
        }
        value.validate_domain(node.domain)?;
        let address = self.input_addresses[&id.index()];
        self.execution_health.invalidate_outputs();
        self.preserved_output_valid.set(false);
        self.execution_health
            .record_io(self.session.copy_to_device(address, value.values()))?;
        self.execution_health
            .record_io(self.session.flush_input_copies())
    }

    /// Establish the static state feedback memory contract before any replay.
    /// Public outputs that alias state inputs receive persistent device
    /// snapshots so step() cannot overwrite observable pure-step results.
    pub(crate) fn configure_model_feedback(
        &mut self,
        public_outputs: &[NodeId],
        links: &[(NodeId, NodeId)],
    ) -> Result<(), GpuError> {
        self.configure_feedback(public_outputs, links, false)
    }

    fn configure_feedback(
        &mut self,
        public_outputs: &[NodeId],
        links: &[(NodeId, NodeId)],
        runtime_domains: bool,
    ) -> Result<(), GpuError> {
        self.execution_health.ensure_ready()?;
        self.feedback_copy_plan(links, runtime_domains)?;
        let destinations = links
            .iter()
            .map(|&(state, _)| self.input_addresses[&state.index()])
            .collect::<BTreeSet<_>>();
        // Internal state roots are not observable to ModelExecution clients.
        let mut checked = BTreeSet::new();
        for &output in public_outputs {
            if !checked.insert(output.index()) {
                return Err(GpuError::InvalidInput {
                    node: output.index(),
                    reason: "duplicate observable feedback output",
                });
            }
            let (shape, address) = self
                .output_addresses
                .iter()
                .find(|(node, _, _)| *node == output)
                .map(|(_, shape, address)| (shape.clone(), *address))
                .ok_or(GpuError::InvalidInput {
                    node: output.index(),
                    reason: "public output is not in the compiled GPU program",
                })?;
            if destinations.contains(&address)
                && !self
                    .preserved_output_addresses
                    .contains_key(&output.index())
            {
                let snapshot = self.session.allocate(shape.element_count())?;
                self.preserved_output_addresses
                    .insert(output.index(), snapshot);
            }
        }
        self.feedback_snapshot_destinations.extend(destinations);
        self.feedback_observable_outputs = Some(public_outputs.to_vec());
        Ok(())
    }

    /// Save the pure step's observable values before writing feedback state.
    /// All copies stay on the owned non-blocking stream; no host round-trip.
    pub(crate) fn snapshot_outputs(&mut self) -> Result<(), GpuError> {
        self.execution_health.ensure_outputs_ready()?;
        self.preserved_output_valid.set(false);
        for (&index, &snapshot) in &self.preserved_output_addresses {
            let (_, shape, source) = self
                .output_addresses
                .iter()
                .find(|(node, _, _)| node.index() == index)
                .expect("configured public output has a live buffer");
            self.execution_health
                .record_io(self.session.copy_device_to_device(
                    snapshot,
                    *source,
                    shape.element_count(),
                ))?;
        }
        if !self.preserved_output_addresses.is_empty() {
            self.execution_health
                .record_io(self.session.synchronize())?;
        }
        self.preserved_output_valid.set(true);
        Ok(())
    }

    fn read_value_at(
        &mut self,
        id: NodeId,
        shape: &Shape,
        address: u64,
    ) -> Result<TensorValue, GpuError> {
        match self.program.graph.node(id)?.dtype {
            DType::F32 => {
                let mut values = vec![0.0f32; shape.element_count()];
                self.execution_health
                    .record_io(self.session.copy_to_host(&mut values, address))?;
                Ok(TensorValue::F32(Tensor::new(shape.clone(), values)?))
            }
            DType::Index => {
                let mut raw = vec![0i32; shape.element_count()];
                self.execution_health
                    .record_io(self.session.copy_indices_to_host(&mut raw, address))?;
                let values = raw
                    .into_iter()
                    .map(|value| {
                        usize::try_from(value).map_err(|_| GpuError::InvalidInput {
                            node: id.index(),
                            reason: "device index output is negative",
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(TensorValue::Index(Tensor::new(shape.clone(), values)?))
            }
        }
    }

    /// Read one selected output while preserving its Semantic IR dtype.
    pub(crate) fn read_selected_value(&mut self, id: NodeId) -> Result<TensorValue, GpuError> {
        self.execution_health.ensure_outputs_ready()?;
        let (_, shape, live) = self
            .output_addresses
            .iter()
            .find(|(node, _, _)| *node == id)
            .cloned()
            .ok_or(GpuError::InvalidInput {
                node: id.index(),
                reason: "requested output is not selected",
            })?;
        let address = if self.preserved_output_valid.get() {
            self.preserved_output_addresses
                .get(&id.index())
                .copied()
                .unwrap_or(live)
        } else {
            live
        };
        self.read_value_at(id, &shape, address)
    }

    /// Backward-compatible F32 reader. Index outputs must use the typed API.
    pub(crate) fn read_selected_output(&mut self, id: NodeId) -> Result<Tensor<f32>, GpuError> {
        match self.read_selected_value(id)? {
            TensorValue::F32(tensor) => Ok(tensor),
            TensorValue::Index(_) => Err(GpuError::InvalidGraph(NeuralError::DTypeMismatch {
                operation: "read_selected_output",
                expected: DType::F32,
                actual: DType::Index,
            })),
        }
    }

    /// Validate feedback shape, dtype, domain, liveness and alias order.
    fn feedback_copy_plan(
        &self,
        links: &[(NodeId, NodeId)],
        runtime_domains: bool,
    ) -> Result<Vec<(u64, u64, usize)>, GpuError> {
        let mut destinations = BTreeSet::new();
        let output_lookup = self
            .output_addresses
            .iter()
            .map(|(id, _, address)| (id.index(), *address))
            .collect::<BTreeMap<_, _>>();
        let mut copies = Vec::with_capacity(links.len());
        for &(state, next) in links {
            let old = self.program.graph.node(state)?;
            let new = self.program.graph.node(next)?;
            if !destinations.insert(state.index()) {
                return Err(GpuError::InvalidInput {
                    node: state.index(),
                    reason: "duplicate state feedback destination",
                });
            }
            if old.shape != new.shape {
                return Err(GpuError::InvalidGraph(NeuralError::ShapeMismatch {
                    expected: old.shape.clone(),
                    actual: new.shape.clone(),
                }));
            }
            if new.dtype != old.dtype {
                return Err(GpuError::InvalidGraph(NeuralError::DTypeMismatch {
                    operation: "feedback_state",
                    expected: old.dtype,
                    actual: new.dtype,
                }));
            }
            if !runtime_domains && !new.domain.satisfies(old.domain) {
                return Err(GpuError::InvalidGraph(NeuralError::DomainNotEstablished {
                    required: old.domain,
                    actual: new.domain,
                }));
            }
            let &dest = self
                .input_addresses
                .get(&state.index())
                .ok_or(GpuError::InvalidInput {
                    node: state.index(),
                    reason: "feedback destination is not a live source",
                })?;
            let &src = output_lookup
                .get(&next.index())
                .ok_or(GpuError::InvalidInput {
                    node: next.index(),
                    reason: "feedback source is not a selected output",
                })?;
            copies.push((dest, src, old.shape.element_count()));
        }
        // Feedback is a parallel, simultaneous assignment. Reject duplicate
        // storage roots and cross-transfer read/write aliasing in O(n log n)
        // rather than comparing every pair of hundreds of optimizer tensors.
        let destinations = copies
            .iter()
            .map(|&(dest, _, _)| dest)
            .collect::<BTreeSet<_>>();
        if destinations.len() != copies.len()
            || copies
                .iter()
                .any(|&(dest, src, _)| dest != src && destinations.contains(&src))
        {
            return Err(GpuError::InvalidInput {
                node: 0,
                reason: "overlapping state feedback sources require staging",
            });
        }
        Ok(copies)
    }

    fn prepared_feedback_plan(
        &mut self,
        links: &[(NodeId, NodeId)],
        runtime_domains: bool,
    ) -> Result<Rc<PreparedFeedback>, GpuError> {
        if let Some(cached) = &self.prepared_feedback {
            if cached.runtime_domains == runtime_domains && cached.links.as_slice() == links {
                return Ok(Rc::clone(cached));
            }
        }
        // A miss must run the original complete preflight before publishing
        // the cache entry or enqueuing a single GPU state transfer.
        let copies = self.feedback_copy_plan(links, runtime_domains)?;
        let mut checked_sources = Vec::new();
        let mut checked_requirements = Vec::new();
        let mut checked_values = 0usize;
        if runtime_domains {
            let output_lookup = self
                .output_addresses
                .iter()
                .map(|(id, _, address)| (id.index(), *address))
                .collect::<BTreeMap<_, _>>();
            for &(state, next) in links {
                let old = self.program.graph.node(state)?;
                let new = self.program.graph.node(next)?;
                if new.domain.satisfies(old.domain) {
                    continue;
                }
                if old.dtype != DType::F32 {
                    return Err(GpuError::InvalidInput {
                        node: state.index(),
                        reason: "runtime feedback domain check requires F32",
                    });
                }
                let elements = old.shape.element_count();
                let &address = output_lookup
                    .get(&next.index())
                    .ok_or(GpuError::InvalidInput {
                        node: next.index(),
                        reason: "feedback source is not a selected output",
                    })?;
                checked_sources.push((address, elements, old.domain == Domain::NonNegativeFinite));
                checked_requirements.push((old.domain, old.shape.clone()));
                checked_values += elements;
            }
        }
        let transfers = copies
            .iter()
            .copied()
            .filter(|(dst, src, _)| dst != src)
            .collect::<Vec<_>>();
        let feedback_values = transfers.iter().map(|&(_, _, n)| n).sum();
        let prepared = Rc::new(PreparedFeedback {
            links: links.to_vec(),
            runtime_domains,
            copies,
            transfers,
            checked_sources,
            checked_requirements,
            checked_values,
            feedback_values,
        });
        self.prepared_feedback = Some(Rc::clone(&prepared));
        Ok(prepared)
    }

    /// Strict feedback for statically established source domains.
    /// Selected outputs aliasing state destinations are device-snapshotted.
    pub fn feedback_state(&mut self, links: &[(NodeId, NodeId)]) -> Result<(), GpuError> {
        self.feedback_state_impl(links, false, false).map(|_| ())
    }

    /// Experimental feedback for outputs with weaker *static* domain
    /// contracts. A separate CUDA Graph checks their FP32 bit patterns and
    /// returns one 32-bit failure flag before **any** state copy. On rejection,
    /// full tensors are read solely to report the exact host-domain error.
    /// No assumption that arbitrary FP32 arithmetic remains finite is
    /// smuggled into IR. The guard is runtime hardware validation, not proof.
    pub fn feedback_state_checked(&mut self, links: &[(NodeId, NodeId)]) -> Result<(), GpuError> {
        self.feedback_state_checked_profiled(links).map(|_| ())
    }

    /// Checked feedback plus stage-separated host timings. Validation is
    /// mandatory and happens before any state mutation, just as in the
    /// unprofiled API. No extra device readback is performed for profiling.
    pub fn feedback_state_checked_profiled(
        &mut self,
        links: &[(NodeId, NodeId)],
    ) -> Result<GpuFeedbackProfile, GpuError> {
        self.feedback_state_impl(links, true, false)
    }

    /// Diagnostic-only variant: measures the copy CUDA Graph with stream
    /// events. This introduces an extra barrier and must never be used to
    /// claim end-to-end speedups. Runtime domain checks remain mandatory.
    pub fn feedback_state_checked_device_profiled(
        &mut self,
        links: &[(NodeId, NodeId)],
    ) -> Result<GpuFeedbackProfile, GpuError> {
        self.feedback_state_impl(links, true, true)
    }

    fn feedback_state_impl(
        &mut self,
        links: &[(NodeId, NodeId)],
        runtime_domains: bool,
        device_events: bool,
    ) -> Result<GpuFeedbackProfile, GpuError> {
        let plan_started = Instant::now();
        self.execution_health.ensure_outputs_ready()?;
        let plan = self.prepared_feedback_plan(links, runtime_domains)?;
        let mut profile = GpuFeedbackProfile {
            plan_host: plan_started.elapsed(),
            checked_values: plan.checked_values,
            feedback_values: plan.feedback_values,
            ..GpuFeedbackProfile::default()
        };
        if plan.copies.is_empty() {
            return Ok(profile);
        }
        if runtime_domains && !plan.checked_sources.is_empty() {
            let validation_started = Instant::now();
            // Descriptors may be cached, but current FP32 output bytes must
            // be checked before every state mutation.
            let domains_valid = self
                .execution_health
                .record_io(self.session.validate_output_domains(&plan.checked_sources))?;
            if !domains_valid {
                // Reject before writing state; exact host oracle is reserved
                // for the exceptional path and retains source order.
                let buffers = plan
                    .checked_sources
                    .iter()
                    .map(|&(address, size, _)| (address, size))
                    .collect::<Vec<_>>();
                let copied = self
                    .execution_health
                    .record_io(self.session.copy_f32_batch_to_host(&buffers))?;
                for ((required, shape), values) in plan.checked_requirements.iter().zip(copied) {
                    Tensor::new(shape.clone(), values)?.validate_domain(*required)?;
                }
                return Err(GpuError::Driver(
                    "GPU domain guard rejected outputs not rejected by the host oracle".into(),
                ));
            }
            profile.validation_d2h = validation_started.elapsed();
        }
        if plan
            .copies
            .iter()
            .any(|&(dst, _, _)| !self.feedback_snapshot_destinations.contains(&dst))
        {
            let observable = self
                .feedback_observable_outputs
                .clone()
                .unwrap_or_else(|| self.program.outputs().to_vec());
            self.configure_feedback(&observable, links, runtime_domains)?;
        }
        let snapshot_started = Instant::now();
        self.snapshot_outputs()?;
        profile.snapshot_d2d = snapshot_started.elapsed();
        let feedback_started = Instant::now();
        if device_events {
            profile.device_copy = Some(
                self.execution_health.record_io(
                    self.session
                        .launch_feedback_copies_device_timed(&plan.transfers),
                )?,
            );
        } else {
            self.execution_health
                .record_io(self.session.launch_feedback_copies(&plan.transfers))?;
        }
        profile.feedback_d2d = feedback_started.elapsed();
        let sync_started = Instant::now();
        self.execution_health
            .record_io(self.session.synchronize())?;
        profile.stream_sync = sync_started.elapsed();
        Ok(profile)
    }

    /// Read a current source (for state inspection) without another replay.
    pub fn read_source(&mut self, id: NodeId) -> Result<Tensor<f32>, GpuError> {
        self.execution_health.ensure_ready()?;
        let node = self.program.graph.node(id)?;
        let address =
            self.input_addresses
                .get(&id.index())
                .copied()
                .ok_or(GpuError::InvalidInput {
                    node: id.index(),
                    reason: "source is outside the compiled output slice",
                })?;
        let mut values = vec![0.0f32; node.shape.element_count()];
        self.execution_health
            .record_io(self.session.copy_to_host(&mut values, address))?;
        Ok(Tensor::new(node.shape.clone(), values)?)
    }

    /// Queue a CUDA Graph replay without synchronizing the stream.
    ///
    /// Other independent executions can enqueue work on their own
    /// non-blocking streams before any synchronization. The outputs remain
    /// inaccessible until `complete_replay` returns successfully. Every
    /// data transfer, feedback and second replay fail closed while pending.
    pub fn enqueue_replay(&self) -> Result<(), GpuError> {
        self.execution_health.begin_replay()?;
        self.preserved_output_valid.set(false);
        if self.has_graph {
            self.execution_health
                .record_io(self.session.launch_graph())?;
        }
        Ok(())
    }

    /// Synchronize this execution's stream and publish the replay outputs.
    pub fn complete_replay(&self) -> Result<(), GpuError> {
        self.execution_health.ensure_pending()?;
        self.execution_health
            .record_io(self.session.synchronize())?;
        self.execution_health.finish_replay()
    }

    /// Legacy synchronous single-execution path, preserving its contract.
    pub fn replay(&self) -> Result<(), GpuError> {
        self.enqueue_replay()?;
        self.complete_replay()
    }

    /// Launch independent CUDA Graph executions on distinct non-blocking
    /// streams before waiting for any of them. No sharing or aliasing of
    /// mutable source buffers between these owned executions is allowed by
    /// their separate CUDA allocations. This is independent inference/
    /// training-task parallelism, **not** gradient accumulation.
    ///
    /// Synchronize every already-launched stream even on an enqueue or
    /// completion error, preventing a stranded pending execution. CUDA
    /// hardware may serialize kernels; concurrency is not guaranteed.
    pub fn replay_independent_batch(executions: &mut [Self]) -> Result<(), GpuError> {
        if executions.is_empty() {
            return Err(GpuError::InvalidInput {
                node: 0,
                reason: "independent GPU execution batch must not be empty",
            });
        }
        // Fail before launching any new work when another member is still
        // pending or has an unrecoverable CUDA failure.
        for execution in executions.iter() {
            execution.execution_health.ensure_ready()?;
        }
        let mut queued = 0usize;
        for execution in executions.iter() {
            if let Err(err) = execution.enqueue_replay() {
                for previous in &executions[..queued] {
                    let _ = previous.complete_replay();
                }
                return Err(err);
            }
            queued += 1;
        }
        let mut first_error = None;
        for execution in executions.iter().take(queued) {
            if let Err(err) = execution.complete_replay() {
                if first_error.is_none() {
                    first_error = Some(err);
                }
            }
        }
        if let Some(err) = first_error {
            Err(err)
        } else {
            Ok(())
        }
    }

    /// Read the primary output while preserving its Semantic IR dtype.
    pub fn read_value(&mut self) -> Result<TensorValue, GpuError> {
        self.execution_health.ensure_outputs_ready()?;
        let id = self.program.output;
        let shape = self.output_shape.clone();
        let address = if self.preserved_output_valid.get() {
            self.preserved_output_addresses
                .get(&id.index())
                .copied()
                .unwrap_or(self.output_address)
        } else {
            self.output_address
        };
        self.read_value_at(id, &shape, address)
    }

    /// Backward-compatible F32 primary-output reader.
    pub fn read_output(&mut self) -> Result<Tensor<f32>, GpuError> {
        match self.read_value()? {
            TensorValue::F32(tensor) => Ok(tensor),
            TensorValue::Index(_) => Err(GpuError::InvalidGraph(NeuralError::DTypeMismatch {
                operation: "read_output",
                expected: DType::F32,
                actual: DType::Index,
            })),
        }
    }

    /// Read each selected output in declaration order, preserving every dtype.
    pub fn read_values(&mut self) -> Result<Vec<(NodeId, TensorValue)>, GpuError> {
        self.execution_health.ensure_outputs_ready()?;
        let outputs = self.output_addresses.clone();
        let mut result = Vec::with_capacity(outputs.len());
        for (node, shape, pointer) in outputs {
            let address = if self.preserved_output_valid.get() {
                self.preserved_output_addresses
                    .get(&node.index())
                    .copied()
                    .unwrap_or(pointer)
            } else {
                pointer
            };
            result.push((node, self.read_value_at(node, &shape, address)?));
        }
        Ok(result)
    }

    /// Read a requested subset of selected F32 outputs in caller order.
    /// All source IDs, duplicate IDs and dtypes are prevalidated before DMA.
    /// One stream barrier completes all queued D2H copies, with pinned staging
    /// retained until then. Results aliasing preserved state snapshots use the
    /// same snapshot selection as read_selected_value().
    pub fn read_outputs_selected(
        &mut self,
        selected: &[NodeId],
    ) -> Result<Vec<(NodeId, Tensor<f32>)>, GpuError> {
        self.read_outputs_selected_profiled(selected)
            .map(|(outputs, _)| outputs)
    }

    /// Identical output semantics to `read_outputs_selected`, additionally
    /// reporting wall-clock host/transfer phases and copied F32 byte count.
    pub fn read_outputs_selected_profiled(
        &mut self,
        selected: &[NodeId],
    ) -> Result<GpuProfiledReadback, GpuError> {
        self.read_outputs_selected_profiled_impl(selected, false)
    }

    /// Experimental: gather all selected outputs through a reusable CUDA
    /// Graph and issue one D2H transfer. No new FP32 computation is performed.
    pub fn read_outputs_selected_packed_profiled(
        &mut self,
        selected: &[NodeId],
    ) -> Result<GpuProfiledReadback, GpuError> {
        self.read_outputs_selected_profiled_impl(selected, true)
    }

    fn read_outputs_selected_profiled_impl(
        &mut self,
        selected: &[NodeId],
        packed: bool,
    ) -> Result<GpuProfiledReadback, GpuError> {
        self.execution_health.ensure_outputs_ready()?;
        let preflight_started = Instant::now();
        let mut seen = BTreeSet::new();
        let mut sources = Vec::with_capacity(selected.len());
        for &id in selected {
            if !seen.insert(id.index()) {
                return Err(GpuError::InvalidInput {
                    node: id.index(),
                    reason: "duplicate selected read output",
                });
            }
            let &slot =
                self.output_address_lookup
                    .get(&id.index())
                    .ok_or(GpuError::InvalidInput {
                        node: id.index(),
                        reason: "requested output is not selected",
                    })?;
            let (_, shape, original) = &self.output_addresses[slot];
            let node = self.program.graph.node(id)?;
            if node.dtype != DType::F32 {
                return Err(GpuError::InvalidGraph(NeuralError::DTypeMismatch {
                    operation: "read_outputs_selected",
                    expected: DType::F32,
                    actual: node.dtype,
                }));
            }
            let address = if self.preserved_output_valid.get() {
                self.preserved_output_addresses
                    .get(&id.index())
                    .copied()
                    .unwrap_or(*original)
            } else {
                *original
            };
            sources.push((id, shape.clone(), address));
        }
        let preflight = preflight_started.elapsed();
        if sources.is_empty() {
            return Ok((
                Vec::new(),
                GpuReadbackProfile {
                    preflight,
                    ..GpuReadbackProfile::default()
                },
            ));
        }
        let buffers = sources
            .iter()
            .map(|(_, shape, address)| (*address, shape.element_count()))
            .collect::<Vec<_>>();
        let transfer = if packed {
            self.session
                .copy_f32_batch_to_host_packed_profiled(&buffers)
        } else {
            self.session.copy_f32_batch_to_host_profiled(&buffers)
        };
        let (copied, timing) = self.execution_health.record_io(transfer)?;
        let construct_started = Instant::now();
        let outputs = sources
            .into_iter()
            .zip(copied)
            .map(|((id, shape, _), values)| Ok((id, Tensor::new(shape, values)?)))
            .collect::<Result<Vec<_>, GpuError>>()?;
        Ok((
            outputs,
            GpuReadbackProfile {
                preflight,
                enqueue: timing.enqueue,
                synchronize: timing.synchronize,
                host_copy: timing.host_copy,
                tensor_construct: construct_started.elapsed(),
                transfers: timing.transfers,
                bytes: timing.bytes,
            },
        ))
    }

    /// Backward-compatible all-F32 reader, now batched in declaration order.
    /// Mixed/Index output sets fail closed.
    pub fn read_outputs(&mut self) -> Result<Vec<(NodeId, Tensor<f32>)>, GpuError> {
        let selected = self.program.outputs().to_vec();
        self.read_outputs_selected(&selected)
    }

    pub fn benchmark(
        &self,
        warmup_launches: usize,
        measured_launches: usize,
    ) -> Result<GpuBenchmark, GpuError> {
        self.execution_health.ensure_ready()?;
        if measured_launches == 0 {
            return Err(GpuError::InvalidBenchmark { iterations: 0 });
        }
        if !self.has_graph {
            return Err(GpuError::NoKernelGraph);
        }
        self.execution_health.invalidate_outputs();
        self.preserved_output_valid.set(false);
        let milliseconds = self.execution_health.record_io(
            self.session
                .benchmark_graph(warmup_launches, measured_launches),
        )?;
        if !milliseconds.is_finite() || milliseconds < 0.0 {
            return Err(GpuError::Driver(
                "CUDA event benchmark returned an invalid duration".into(),
            ));
        }
        // benchmark_graph synchronizes the stop event, so the final measured
        // replay has completed and its output buffers are safe to read.
        self.execution_health.outputs_completed();
        Ok(GpuBenchmark {
            warmup_launches,
            measured_launches,
            elapsed: Duration::from_secs_f64(f64::from(milliseconds) / 1000.0),
        })
    }
}

