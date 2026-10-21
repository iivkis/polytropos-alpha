//! CUDA Driver API bindings loaded from the system NVIDIA driver at runtime.
//! ABI bindings here are deliberately limited to the stage-4 synchronous path.

#[cfg(target_os = "linux")]
mod platform {
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
        context_create: unsafe extern "C" fn(*mut Handle, c_uint, c_int) -> ResultCode,
        context_destroy: unsafe extern "C" fn(Handle) -> ResultCode,
        module_load: unsafe extern "C" fn(*mut Handle, *const c_void) -> ResultCode,
        module_unload: unsafe extern "C" fn(Handle) -> ResultCode,
        module_function: unsafe extern "C" fn(*mut Handle, Handle, *const c_char) -> ResultCode,
        mem_allocate: unsafe extern "C" fn(*mut DevicePtr, usize) -> ResultCode,
        mem_free: unsafe extern "C" fn(DevicePtr) -> ResultCode,
        host_to_device: unsafe extern "C" fn(DevicePtr, *const c_void, usize) -> ResultCode,
        device_to_host: unsafe extern "C" fn(*mut c_void, DevicePtr, usize) -> ResultCode,
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
        synchronize: unsafe extern "C" fn() -> ResultCode,
    }

    impl Functions {
        fn load(lib: &Library) -> Result<Self, GpuError> {
            Ok(Self {
                init: lib.function("cuInit")?,
                device_get: lib.function("cuDeviceGet")?,
                context_create: lib.function("cuCtxCreate_v2")?,
                context_destroy: lib.function("cuCtxDestroy_v2")?,
                module_load: lib.function("cuModuleLoadData")?,
                module_unload: lib.function("cuModuleUnload")?,
                module_function: lib.function("cuModuleGetFunction")?,
                mem_allocate: lib.function("cuMemAlloc_v2")?,
                mem_free: lib.function("cuMemFree_v2")?,
                host_to_device: lib.function("cuMemcpyHtoD_v2")?,
                device_to_host: lib.function("cuMemcpyDtoH_v2")?,
                launch: lib.function("cuLaunchKernel")?,
                synchronize: lib.function("cuCtxSynchronize")?,
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

    /// Owns every device allocation, loaded PTX module and CUDA context.
    /// The library is dropped last, after all CUDA handles are released.
    pub(crate) struct CudaSession {
        functions: Functions,
        context: Handle,
        module: Handle,
        allocations: Vec<DevicePtr>,
        _library: Library,
    }

    impl CudaSession {
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
            let mut context = ptr::null_mut();
            // SAFETY: context is a valid output pointer; CUDA owns the context.
            check("cuCtxCreate_v2", unsafe {
                (functions.context_create)(&mut context, 0, device)
            })?;
            let mut session = Self {
                functions,
                context,
                module: ptr::null_mut(),
                allocations: Vec::new(),
                _library: library,
            };
            let bytes = CString::new(ptx)
                .map_err(|_| GpuError::Driver("PTX contains interior NUL byte".into()))?;
            // SAFETY: PTX has an extra terminating NUL and lives through the call.
            check("cuModuleLoadData", unsafe {
                (session.functions.module_load)(
                    &mut session.module,
                    bytes.as_ptr().cast::<c_void>(),
                )
            })?;
            Ok(session)
        }

        pub fn allocate(&mut self, elements: usize) -> Result<DevicePtr, GpuError> {
            let bytes = elements
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(GpuError::UnsupportedSize { elements })?;
            let mut address = 0;
            // SAFETY: address is writable, CUDA manages allocated device memory.
            check("cuMemAlloc_v2", unsafe {
                (self.functions.mem_allocate)(&mut address, bytes)
            })?;
            self.allocations.push(address);
            Ok(address)
        }

        pub fn copy_to_device(&self, address: DevicePtr, values: &[f32]) -> Result<(), GpuError> {
            let bytes = std::mem::size_of_val(values);
            // SAFETY: address is a valid allocation owned by this session and the
            // source slice remains alive for the synchronous CUDA copy.
            check("cuMemcpyHtoD_v2", unsafe {
                (self.functions.host_to_device)(address, values.as_ptr().cast::<c_void>(), bytes)
            })
        }

        pub fn copy_to_host(&self, out: &mut [f32], address: DevicePtr) -> Result<(), GpuError> {
            let bytes = std::mem::size_of_val(out);
            // SAFETY: destination slice is writable and device allocation is
            // valid; the copy is synchronous after cuCtxSynchronize.
            check("cuMemcpyDtoH_v2", unsafe {
                (self.functions.device_to_host)(out.as_mut_ptr().cast::<c_void>(), address, bytes)
            })
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
            let name = CString::new(name)
                .map_err(|_| GpuError::Driver("kernel name contains NUL".into()))?;
            let mut function = ptr::null_mut();
            // SAFETY: name is NUL-terminated and module remains loaded.
            check("cuModuleGetFunction", unsafe {
                (self.functions.module_function)(&mut function, self.module, name.as_ptr())
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
                    ptr::null_mut(),
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                )
            })
        }

        pub fn synchronize(&self) -> Result<(), GpuError> {
            // SAFETY: session context is current for the calling host thread.
            check("cuCtxSynchronize", unsafe {
                (self.functions.synchronize)()
            })
        }
    }

    impl Drop for CudaSession {
        fn drop(&mut self) {
            // SAFETY: these handles belong to this session. Attempt to release
            // all resources even when the preceding operation has failed.
            for address in self.allocations.drain(..).rev() {
                unsafe { (self.functions.mem_free)(address) };
            }
            if !self.module.is_null() {
                unsafe { (self.functions.module_unload)(self.module) };
            }
            if !self.context.is_null() {
                unsafe { (self.functions.context_destroy)(self.context) };
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

        pub fn copy_to_device(&self, _: u64, _: &[f32]) -> Result<(), GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn copy_to_host(&self, _: &mut [f32], _: u64) -> Result<(), GpuError> {
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

        pub fn synchronize(&self) -> Result<(), GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }
    }
}

pub(super) use platform::CudaSession;
