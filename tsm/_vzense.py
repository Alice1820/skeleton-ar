import ctypes
import sys
import traceback


_library_handle = None

class vzense:

	def __init__(self):

		dll = _library_handle


		"""
		K4A_EXPORT k4a_result_t k4a_set_allocator(k4a_memory_allocate_cb_t allocate, k4a_memory_destroy_cb_t free);                                                                                        
		"""

        #
		#K4A_EXPORT k4a_result_t k4a_device_open(uint32_t index, k4a_device_t *device_handle);
		self.k4a_device_open = dll.k4a_device_open
		self.k4a_device_open.restype=ctypes.c_int
		self.k4a_device_open.argtypes=(ctypes.c_uint32, ctypes.POINTER(k4a_device_t))

		#K4A_EXPORT void k4a_device_close(k4a_device_t device_handle);
		self.k4a_device_close = dll.k4a_device_close
		self.k4a_device_close.restype=None
		self.k4a_device_close.argtypes=(k4a_device_t,)