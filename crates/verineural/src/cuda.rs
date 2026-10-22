//! CUDA Driver API bindings loaded from the system NVIDIA driver at runtime.
//! GPU kernels execute only as CUDA Graph nodes on an explicit non-blocking stream.
//! Legacy/default streams, direct standalone kernel launches and silent fallbacks
//! are forbidden. Host transfers also use this explicit stream.

#[cfg(target_os = "linux")]
mod platform {
    use std::collections::BTreeMap;
    use std::ffi::{CString, c_char, c_int, c_uint, c_void};
    use std::mem::transmute_copy;
    use std::ptr;

    use super::super::gpu::GpuError;

    #[link(name = "dl")]
    unsafe extern "C" {
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> c_int;
    }

    const RTLD_NOW: c_int = 2;
    const CU_STREAM_NON_BLOCKING: c_uint = 1;
    const CU_STREAM_CAPTURE_MODE_THREAD_LOCAL: c_int = 1;
    const CU_EVENT_DEFAULT: c_uint = 0;

    struct Library(*mut c_void);

    impl Library {
        fn open() -> Result<Self, GpuError> {
            let name = c"libcuda.so.1";
            // SAFETY: name is a NUL-terminated static string.
            let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
            if handle.is_null() {
                return Err(GpuError::Driver(
                    "could not load libcuda.so.1; install an NVIDIA driver".into(),
                ));
            }
            Ok(Self(handle))
        }

        fn function<T: Copy>(&self, name: &'static str) -> Result<T, GpuError> {
            let cname = CString::new(name)
                .map_err(|_| GpuError::Driver("invalid CUDA symbol name".into()))?;
            // SAFETY: handle is live and cname is NUL-terminated.
            let symbol = unsafe { dlsym(self.0, cname.as_ptr()) };
            if symbol.is_null() {
                return Err(GpuError::Driver(format!("missing CUDA symbol {name}")));
            }
            if std::mem::size_of::<T>() != std::mem::size_of::<*mut c_void>() {
                return Err(GpuError::Driver(
                    "CUDA symbol has an invalid ABI size".into(),
                ));
            }
            // SAFETY: each caller chooses T to match the exact CUDA C ABI.
            Ok(unsafe { transmute_copy(&symbol) })
        }
    }

    impl Drop for Library {
        fn drop(&mut self) {
            // SAFETY: the handle was returned by dlopen and is owned by us.
            unsafe { dlclose(self.0) };
        }
    }

    type Handle = *mut c_void;
    type DevicePtr = u64;
    type ResultCode = c_int;

    struct Functions {
        init: unsafe extern "C" fn(c_uint) -> ResultCode,
        device_get: unsafe extern "C" fn(*mut c_int, c_int) -> ResultCode,
        primary_retain: unsafe extern "C" fn(*mut Handle, c_int) -> ResultCode,
        primary_release: unsafe extern "C" fn(c_int) -> ResultCode,
        context_push: unsafe extern "C" fn(Handle) -> ResultCode,
        context_pop: unsafe extern "C" fn(*mut Handle) -> ResultCode,
        module_load: unsafe extern "C" fn(*mut Handle, *const c_void) -> ResultCode,
        module_unload: unsafe extern "C" fn(Handle) -> ResultCode,
        module_function: unsafe extern "C" fn(*mut Handle, Handle, *const c_char) -> ResultCode,
        mem_allocate: unsafe extern "C" fn(*mut DevicePtr, usize) -> ResultCode,
        mem_free: unsafe extern "C" fn(DevicePtr) -> ResultCode,
        pinned_allocate: unsafe extern "C" fn(*mut *mut c_void, usize) -> ResultCode,
        pinned_free: unsafe extern "C" fn(*mut c_void) -> ResultCode,
        host_to_device: unsafe extern "C" fn(DevicePtr, *const c_void, usize, Handle) -> ResultCode,
        device_to_host: unsafe extern "C" fn(*mut c_void, DevicePtr, usize, Handle) -> ResultCode,
        stream_create: unsafe extern "C" fn(*mut Handle, c_uint) -> ResultCode,
        stream_destroy: unsafe extern "C" fn(Handle) -> ResultCode,
        stream_sync: unsafe extern "C" fn(Handle) -> ResultCode,
        stream_capture_begin: unsafe extern "C" fn(Handle, c_int) -> ResultCode,
        stream_capture_end: unsafe extern "C" fn(Handle, *mut Handle) -> ResultCode,
        graph_instantiate: unsafe extern "C" fn(*mut Handle, Handle, u64) -> ResultCode,
        graph_launch: unsafe extern "C" fn(Handle, Handle) -> ResultCode,
        graph_destroy: unsafe extern "C" fn(Handle) -> ResultCode,
        graph_exec_destroy: unsafe extern "C" fn(Handle) -> ResultCode,
        event_create: unsafe extern "C" fn(*mut Handle, c_uint) -> ResultCode,
        event_record: unsafe extern "C" fn(Handle, Handle) -> ResultCode,
        event_sync: unsafe extern "C" fn(Handle) -> ResultCode,
        event_elapsed: unsafe extern "C" fn(*mut f32, Handle, Handle) -> ResultCode,
        event_destroy: unsafe extern "C" fn(Handle) -> ResultCode,
        launch: unsafe extern "C" fn(
            Handle,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            Handle,
            *mut *mut c_void,
            *mut *mut c_void,
        ) -> ResultCode,
    }

    impl Functions {
        fn load(lib: &Library) -> Result<Self, GpuError> {
            Ok(Self {
                init: lib.function("cuInit")?,
                device_get: lib.function("cuDeviceGet")?,
                primary_retain: lib.function("cuDevicePrimaryCtxRetain")?,
                primary_release: lib.function("cuDevicePrimaryCtxRelease")?,
                context_push: lib.function("cuCtxPushCurrent_v2")?,
                context_pop: lib.function("cuCtxPopCurrent_v2")?,
                module_load: lib.function("cuModuleLoadData")?,
                module_unload: lib.function("cuModuleUnload")?,
                module_function: lib.function("cuModuleGetFunction")?,
                mem_allocate: lib.function("cuMemAlloc_v2")?,
                mem_free: lib.function("cuMemFree_v2")?,
                pinned_allocate: lib.function("cuMemAllocHost_v2")?,
                pinned_free: lib.function("cuMemFreeHost")?,
                host_to_device: lib.function("cuMemcpyHtoDAsync_v2")?,
                device_to_host: lib.function("cuMemcpyDtoHAsync_v2")?,
                stream_create: lib.function("cuStreamCreate")?,
                stream_destroy: lib.function("cuStreamDestroy_v2")?,
                stream_sync: lib.function("cuStreamSynchronize")?,
                // CUDA Driver API: the unversioned cuStreamBeginCapture takes ONE
                // argument. We need the two-argument v2 ABI (mode included).
                stream_capture_begin: lib.function("cuStreamBeginCapture_v2")?,
                stream_capture_end: lib.function("cuStreamEndCapture")?,
                graph_instantiate: lib.function("cuGraphInstantiateWithFlags")?,
                graph_launch: lib.function("cuGraphLaunch")?,
                graph_destroy: lib.function("cuGraphDestroy")?,
                graph_exec_destroy: lib.function("cuGraphExecDestroy")?,
                event_create: lib.function("cuEventCreate")?,
                event_record: lib.function("cuEventRecord")?,
                event_sync: lib.function("cuEventSynchronize")?,
                event_elapsed: lib.function("cuEventElapsedTime")?,
                event_destroy: lib.function("cuEventDestroy_v2")?,
                launch: lib.function("cuLaunchKernel")?,
            })
        }
    }

    fn check(function: &'static str, code: ResultCode) -> Result<(), GpuError> {
        if code == 0 {
            Ok(())
        } else {
            Err(GpuError::Cuda { function, code })
        }
    }

    struct ContextGuard {
        pop: unsafe extern "C" fn(*mut Handle) -> ResultCode,
        expected: Handle,
    }

    impl Drop for ContextGuard {
        fn drop(&mut self) {
            let mut popped = ptr::null_mut();
            // SAFETY: this guard exists only after a successful push on this thread.
            let code = unsafe { (self.pop)(&mut popped) };
            debug_assert_eq!(code, 0);
            debug_assert_eq!(popped, self.expected);
        }
    }

    // A pinned allocation can be reused only after its previous async
    // transfer has completed on the session's explicit stream.
    struct PinnedBuffer {
        address: *mut c_void,
        capacity: usize,
    }

    /// Owns the stream, PTX module, device allocations and a retained reference
    /// to the shared primary CUDA context. Each session retains/releases once.
    /// The library is dropped last, after all CUDA handles are released.
    pub(crate) struct CudaSession {
        functions: Functions,
        device: c_int,
        context: Handle,
        module: Handle,
        allocations: Vec<(DevicePtr, usize)>,
        staging_pending: Vec<PinnedBuffer>,
        staging_available: Vec<PinnedBuffer>,
        kernels: BTreeMap<String, Handle>,
        stream: Handle,
        graph: Handle,
        graph_exec: Handle,
        capturing: bool,
        _library: Library,
    }

    impl CudaSession {
        fn enter_context(&self) -> Result<ContextGuard, GpuError> {
            // SAFETY: the context is owned by this session and remains live.
            check("cuCtxPushCurrent_v2", unsafe {
                (self.functions.context_push)(self.context)
            })?;
            Ok(ContextGuard {
                pop: self.functions.context_pop,
                expected: self.context,
            })
        }

        fn pop_context(&self) -> Result<(), GpuError> {
            let mut popped = ptr::null_mut();
            // SAFETY: callers only pop a context previously pushed on this thread.
            check("cuCtxPopCurrent_v2", unsafe {
                (self.functions.context_pop)(&mut popped)
            })?;
            if popped != self.context {
                return Err(GpuError::Driver(
                    "CUDA popped a different context than the session owns".into(),
                ));
            }
            Ok(())
        }

        pub fn new(ptx: &[u8]) -> Result<Self, GpuError> {
            let library = Library::open()?;
            let functions = Functions::load(&library)?;
            // SAFETY: CUDA has been dynamically loaded with matching ABI.
            check("cuInit", unsafe { (functions.init)(0) })?;
            let mut device = 0;
            // SAFETY: device is a valid writable pointer.
            check("cuDeviceGet", unsafe {
                (functions.device_get)(&mut device, 0)
            })?;
            // The primary context is shared by all sessions on this device.
            // Unlike cuCtxCreate, retain does not push it on this thread.
            let mut context = ptr::null_mut();
            check("cuDevicePrimaryCtxRetain", unsafe {
                (functions.primary_retain)(&mut context, device)
            })?;
            if context.is_null() {
                // A successful retain must be balanced even for a bad handle.
                unsafe { (functions.primary_release)(device) };
                return Err(GpuError::Driver(
                    "CUDA returned a null primary context".into(),
                ));
            }
            let mut session = Self {
                functions,
                device,
                context,
                module: ptr::null_mut(),
                allocations: Vec::new(),
                staging_pending: Vec::new(),
                staging_available: Vec::new(),
                kernels: BTreeMap::new(),
                stream: ptr::null_mut(),
                graph: ptr::null_mut(),
                graph_exec: ptr::null_mut(),
                capturing: false,
                _library: library,
            };
            // This guard is dropped before session teardown on any error.
            let _context = session.enter_context()?;
            let bytes = CString::new(ptx)
                .map_err(|_| GpuError::Driver("PTX contains interior NUL byte".into()))?;
            // SAFETY: PTX has an extra terminating NUL and lives through the call.
            check("cuModuleLoadData", unsafe {
                (session.functions.module_load)(
                    &mut session.module,
                    bytes.as_ptr().cast::<c_void>(),
                )
            })?;
            // Never pass 0/CU_STREAM_LEGACY/CU_STREAM_PER_THREAD as a launch stream.
            // No fallback: require an explicitly allocated non-blocking stream.
            check("cuStreamCreate", unsafe {
                (session.functions.stream_create)(&mut session.stream, CU_STREAM_NON_BLOCKING)
            })?;
            if session.stream.is_null() {
                return Err(GpuError::Driver(
                    "CUDA returned a null stream handle".into(),
                ));
            }

            // _context pops the retained primary context after setup. Sessions
            // share a device context but never share stream/module ownership.
            Ok(session)
        }

        pub fn allocate(&mut self, elements: usize) -> Result<DevicePtr, GpuError> {
            let _context = self.enter_context()?;
            let bytes = elements
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(GpuError::UnsupportedSize { elements })?;
            let mut address = 0;
            // SAFETY: address is writable, CUDA manages allocated device memory.
            check("cuMemAlloc_v2", unsafe {
                (self.functions.mem_allocate)(&mut address, bytes)
            })?;
            self.allocations.push((address, elements));
            Ok(address)
        }

        /// Check the actual device allocations backing a logical memory
        /// contract before graph capture. This is a preflight check, not an
        /// assertion about LLVM/PTX's machine-level memory accesses.
        pub fn require_allocation(
            &self,
            address: DevicePtr,
            elements: usize,
        ) -> Result<(), GpuError> {
            if !self
                .allocations
                .iter()
                .any(|&(ptr, capacity)| ptr == address && capacity >= elements)
            {
                return Err(GpuError::Driver(
                    "CUDA kernel memory contract references an unallocated or undersized buffer"
                        .into(),
                ));
            }
            Ok(())
        }

        // Find the smallest completed allocation that fits, or allocate one.
        // Never recycle in-flight buffers: async CUDA copies still read them.
        fn staging_buffer(&mut self, bytes: usize) -> Result<PinnedBuffer, GpuError> {
            let best = self
                .staging_available
                .iter()
                .enumerate()
                .filter(|(_, entry)| entry.capacity >= bytes)
                .min_by_key(|(_, entry)| entry.capacity)
                .map(|(index, _)| index);
            if let Some(index) = best {
                return Ok(self.staging_available.swap_remove(index));
            }
            let mut address = ptr::null_mut();
            check("cuMemAllocHost_v2", unsafe {
                (self.functions.pinned_allocate)(&mut address, bytes)
            })?;
            if address.is_null() {
                return Err(GpuError::Driver(
                    "CUDA returned null pinned allocation".into(),
                ));
            }
            Ok(PinnedBuffer {
                address,
                capacity: bytes,
            })
        }

        /// Stage input bytes in page-locked memory and enqueue on our explicit
        /// non-blocking stream. No buffer is recycled before synchronization.
        pub fn copy_to_device(
            &mut self,
            address: DevicePtr,
            values: &[f32],
        ) -> Result<(), GpuError> {
            let _context = self.enter_context()?;
            let bytes = std::mem::size_of_val(values);
            let staging = self.staging_buffer(bytes)?;
            // SAFETY: staging has capacity for values.len() f32s.
            unsafe {
                ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    staging.address.cast::<f32>(),
                    values.len(),
                )
            };
            // Keep the staging allocation alive even if enqueueing fails:
            // CUDA may have observed it before returning an error.
            let pinned = staging.address;
            self.staging_pending.push(staging);
            check("cuMemcpyHtoDAsync_v2", unsafe {
                (self.functions.host_to_device)(address, pinned, bytes, self.stream)
            })
        }

        pub fn copy_to_host(
            &mut self,
            out: &mut [f32],
            address: DevicePtr,
        ) -> Result<(), GpuError> {
            let _context = self.enter_context()?;
            let bytes = std::mem::size_of_val(out);
            let staging = self.staging_buffer(bytes)?;
            let pinned = staging.address;
            self.staging_pending.push(staging);
            // SAFETY: destination remains valid through synchronization.
            check("cuMemcpyDtoHAsync_v2", unsafe {
                (self.functions.device_to_host)(pinned, address, bytes, self.stream)
            })?;
            self.synchronize()?;
            // SAFETY: the async transfer completed; Rust out is valid for out.len() f32s.
            unsafe { ptr::copy_nonoverlapping(pinned.cast::<f32>(), out.as_mut_ptr(), out.len()) };
            self.release_completed_staging();
            Ok(())
        }

        // Reuse completed staging buffers instead of allocating/freeing pinned
        // memory on every update_inputs/read_output. The session owns both lists.
        fn release_completed_staging(&mut self) {
            self.staging_available.append(&mut self.staging_pending);
        }

        pub fn flush_input_copies(&mut self) -> Result<(), GpuError> {
            self.synchronize()?;
            self.release_completed_staging();
            Ok(())
        }

        /// Begin capturing only kernel launches to a CUDA Graph. No allocations,
        /// host copies, or default-stream commands are permitted during capture.
        pub fn begin_graph(&mut self) -> Result<(), GpuError> {
            if self.capturing || !self.graph.is_null() || !self.graph_exec.is_null() {
                return Err(GpuError::Driver("CUDA Graph already in progress".into()));
            }
            self.flush_input_copies()?;
            // Keep this session context current for the entire capture sequence.
            check("cuCtxPushCurrent_v2", unsafe {
                (self.functions.context_push)(self.context)
            })?;
            // SAFETY: an explicitly created nonblocking stream is current.
            let code = unsafe {
                (self.functions.stream_capture_begin)(
                    self.stream,
                    CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )
            };
            if let Err(err) = check("cuStreamBeginCapture", code) {
                let _ = self.pop_context();
                return Err(err);
            }
            self.capturing = true;
            Ok(())
        }

        /// Finish capture and instantiate an executable graph without launching it.
        pub fn finish_graph(&mut self) -> Result<(), GpuError> {
            if !self.capturing {
                return Err(GpuError::Driver("CUDA Graph was not started".into()));
            }
            let result = (|| {
                let mut graph = ptr::null_mut();
                // SAFETY: capture runs on the current thread and same CUDA stream.
                let code = unsafe { (self.functions.stream_capture_end)(self.stream, &mut graph) };
                self.capturing = false;
                self.graph = graph;
                check("cuStreamEndCapture", code)?;
                if self.graph.is_null() {
                    return Err(GpuError::Driver("CUDA Graph capture returned null".into()));
                }
                // SAFETY: all captured kernel parameters have been copied by CUDA.
                check("cuGraphInstantiateWithFlags", unsafe {
                    (self.functions.graph_instantiate)(&mut self.graph_exec, self.graph, 0)
                })
            })();
            let pop_result = self.pop_context();
            result?;
            pop_result
        }

        /// Replay an already instantiated graph on the owned non-blocking stream.
        pub fn launch_graph(&self) -> Result<(), GpuError> {
            let _context = self.enter_context()?;
            self.launch_graph_current_context()
        }

        // For use inside a longer-lived context guard (CUDA-event benchmarks).
        fn launch_graph_current_context(&self) -> Result<(), GpuError> {
            if self.capturing || self.graph_exec.is_null() {
                return Err(GpuError::Driver(
                    "CUDA Graph must be instantiated before replay".into(),
                ));
            }
            check("cuGraphLaunch", unsafe {
                (self.functions.graph_launch)(self.graph_exec, self.stream)
            })
        }

        /// Measure graph replay on the CUDA stream with device events.
        pub fn benchmark_graph(
            &self,
            warmup_launches: usize,
            measured_launches: usize,
        ) -> Result<f32, GpuError> {
            let _context = self.enter_context()?;
            for _ in 0..warmup_launches {
                self.launch_graph_current_context()?;
            }
            self.synchronize()?;

            let mut start = ptr::null_mut();
            check("cuEventCreate", unsafe {
                (self.functions.event_create)(&mut start, CU_EVENT_DEFAULT)
            })?;
            let mut stop = ptr::null_mut();
            if let Err(err) = check("cuEventCreate", unsafe {
                (self.functions.event_create)(&mut stop, CU_EVENT_DEFAULT)
            }) {
                unsafe { (self.functions.event_destroy)(start) };
                return Err(err);
            }

            let result = (|| {
                check("cuEventRecord", unsafe {
                    (self.functions.event_record)(start, self.stream)
                })?;
                for _ in 0..measured_launches {
                    self.launch_graph_current_context()?;
                }
                check("cuEventRecord", unsafe {
                    (self.functions.event_record)(stop, self.stream)
                })?;
                check("cuEventSynchronize", unsafe {
                    (self.functions.event_sync)(stop)
                })?;
                let mut milliseconds = 0.0f32;
                check("cuEventElapsedTime", unsafe {
                    (self.functions.event_elapsed)(&mut milliseconds, start, stop)
                })?;
                Ok(milliseconds)
            })();

            unsafe {
                (self.functions.event_destroy)(stop);
                (self.functions.event_destroy)(start);
            }
            result
        }

        /// Resolve the CUDA function before capture. Lazy lookup/JIT is not
        /// part of the captured stream's execution dependency graph.
        pub fn prepare_kernel(&mut self, name: &str) -> Result<(), GpuError> {
            let _context = self.enter_context()?;
            if self.capturing {
                return Err(GpuError::Driver(
                    "kernel resolution during CUDA Graph capture is forbidden".into(),
                ));
            }
            if self.kernels.contains_key(name) {
                return Ok(());
            }
            let c_name = CString::new(name)
                .map_err(|_| GpuError::Driver("kernel name contains NUL".into()))?;
            let mut function = ptr::null_mut();
            // SAFETY: module is loaded and the C string is NUL terminated.
            check("cuModuleGetFunction", unsafe {
                (self.functions.module_function)(&mut function, self.module, c_name.as_ptr())
            })?;
            self.kernels.insert(name.to_owned(), function);
            Ok(())
        }

        pub fn launch(
            &self,
            name: &str,
            left: DevicePtr,
            right: DevicePtr,
            output: DevicePtr,
            length: usize,
            threads: u32,
        ) -> Result<(), GpuError> {
            if !self.capturing {
                return Err(GpuError::Driver(
                    "direct kernel launch is forbidden; capture a CUDA Graph".into(),
                ));
            }
            let function = *self.kernels.get(name).ok_or_else(|| {
                GpuError::Driver(
                    "kernel function must be resolved before CUDA Graph capture".into(),
                )
            })?;
            let count = i32::try_from(length)
                .map_err(|_| GpuError::UnsupportedSize { elements: length })?;
            let grid = u32::try_from(length.div_ceil(threads as usize))
                .map_err(|_| GpuError::UnsupportedSize { elements: length })?;
            let mut lhs = left;
            let mut rhs = right;
            let mut dst = output;
            let mut n = count;
            let mut args = [
                (&mut lhs as *mut DevicePtr).cast::<c_void>(),
                (&mut rhs as *mut DevicePtr).cast::<c_void>(),
                (&mut dst as *mut DevicePtr).cast::<c_void>(),
                (&mut n as *mut i32).cast::<c_void>(),
            ];
            // SAFETY: each kernel argument points to correctly sized stable
            // storage matching the LLVM kernel signature. The launch itself
            // is asynchronous; synchronize before copying results.
            check("cuLaunchKernel", unsafe {
                (self.functions.launch)(
                    function,
                    grid,
                    1,
                    1,
                    threads,
                    1,
                    1,
                    0,
                    self.stream,
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                )
            })
        }

        pub fn launch3(
            &self,
            name: &str,
            sources: [DevicePtr; 3],
            output: DevicePtr,
            length: usize,
            threads: u32,
        ) -> Result<(), GpuError> {
            if !self.capturing {
                return Err(GpuError::Driver(
                    "direct kernel launch is forbidden; capture a CUDA Graph".into(),
                ));
            }
            let function = *self.kernels.get(name).ok_or_else(|| {
                GpuError::Driver(
                    "kernel function must be resolved before CUDA Graph capture".into(),
                )
            })?;
            let count = i32::try_from(length)
                .map_err(|_| GpuError::UnsupportedSize { elements: length })?;
            let grid = u32::try_from(length.div_ceil(threads as usize))
                .map_err(|_| GpuError::UnsupportedSize { elements: length })?;
            let [mut a, mut b, mut c] = sources;
            let mut dst = output;
            let mut n = count;
            let mut args = [
                (&mut a as *mut DevicePtr).cast::<c_void>(),
                (&mut b as *mut DevicePtr).cast::<c_void>(),
                (&mut c as *mut DevicePtr).cast::<c_void>(),
                (&mut dst as *mut DevicePtr).cast::<c_void>(),
                (&mut n as *mut i32).cast::<c_void>(),
            ];
            check("cuLaunchKernel", unsafe {
                (self.functions.launch)(
                    function,
                    grid,
                    1,
                    1,
                    threads,
                    1,
                    1,
                    0,
                    self.stream,
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                )
            })
        }

        pub fn synchronize(&self) -> Result<(), GpuError> {
            let _context = self.enter_context()?;
            // SAFETY: stream belongs to this CUDA context; never synchronize legacy.
            check("cuStreamSynchronize", unsafe {
                (self.functions.stream_sync)(self.stream)
            })
        }
    }

    impl Drop for CudaSession {
        fn drop(&mut self) {
            // begin_graph intentionally keeps the context pushed until capture ends.
            let mut context_pushed = self.capturing;
            if !context_pushed && !self.context.is_null() {
                context_pushed = unsafe { (self.functions.context_push)(self.context) } == 0;
            }

            // The stream must complete before releasing graph/module/device/pinned
            // buffers. A failed capture is ended best-effort without replay.
            if self.capturing && !self.stream.is_null() {
                let mut graph = ptr::null_mut();
                // SAFETY: same CUDA stream, no concurrent host access.
                unsafe { (self.functions.stream_capture_end)(self.stream, &mut graph) };
                self.capturing = false;
                if !graph.is_null() {
                    unsafe { (self.functions.graph_destroy)(graph) };
                }
            }
            if !self.stream.is_null() {
                // SAFETY: session owns this stream.
                unsafe { (self.functions.stream_sync)(self.stream) };
            }
            if !self.graph_exec.is_null() {
                // SAFETY: the executable graph is owned by this session.
                unsafe { (self.functions.graph_exec_destroy)(self.graph_exec) };
            }
            if !self.graph.is_null() {
                unsafe { (self.functions.graph_destroy)(self.graph) };
            }
            for (address, _) in self.allocations.drain(..).rev() {
                unsafe { (self.functions.mem_free)(address) };
            }
            for host in self
                .staging_pending
                .drain(..)
                .chain(self.staging_available.drain(..))
            {
                unsafe { (self.functions.pinned_free)(host.address) };
            }
            if !self.module.is_null() {
                unsafe { (self.functions.module_unload)(self.module) };
            }
            if !self.stream.is_null() {
                unsafe { (self.functions.stream_destroy)(self.stream) };
            }
            if context_pushed {
                let mut popped = ptr::null_mut();
                unsafe { (self.functions.context_pop)(&mut popped) };
                debug_assert_eq!(popped, self.context);
            }
            // Release exactly one primary-context reference *after* destroying
            // every per-session CUDA object and popping the current context.
            if !self.context.is_null() {
                unsafe { (self.functions.primary_release)(self.device) };
            }
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod platform {
    use super::super::gpu::GpuError;

    pub(crate) struct CudaSession;

    impl CudaSession {
        pub fn new(_: &[u8]) -> Result<Self, GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn allocate(&mut self, _: usize) -> Result<u64, GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn require_allocation(&self, _: u64, _: usize) -> Result<(), GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn copy_to_device(&mut self, _: u64, _: &[f32]) -> Result<(), GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn copy_to_host(&mut self, _: &mut [f32], _: u64) -> Result<(), GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn begin_graph(&mut self) -> Result<(), GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn finish_graph(&mut self) -> Result<(), GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn launch_graph(&self) -> Result<(), GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn benchmark_graph(&self, _: usize, _: usize) -> Result<f32, GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn prepare_kernel(&mut self, _: &str) -> Result<(), GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn launch(
            &self,
            _: &str,
            _: u64,
            _: u64,
            _: u64,
            _: usize,
            _: u32,
        ) -> Result<(), GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn launch3(
            &self,
            _: &str,
            _: [u64; 3],
            _: u64,
            _: usize,
            _: u32,
        ) -> Result<(), GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn synchronize(&self) -> Result<(), GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn flush_input_copies(&mut self) -> Result<(), GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }
    }
}

pub(super) use platform::CudaSession;
