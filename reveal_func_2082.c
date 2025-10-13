static int udev_device_info ( struct libusb_context * ctx , int detached , struct udev_device * udev_dev , uint8_t * busnum , uint8_t * devaddr , const char * * sys_name ) {
 const char * dev_node ;
 dev_node = udev_device_get_devnode ( udev_dev ) ;
 if ( ! dev_node ) {
 return LIBUSB_ERROR_OTHER ;
 }
 * sys_name = udev_device_get_sysname ( udev_dev ) ;
 if ( ! * sys_name ) {
 return LIBUSB_ERROR_OTHER ;
 }
 return linux_get_device_address ( ctx , detached , busnum , devaddr , dev_node , * sys_name ) ;
 }