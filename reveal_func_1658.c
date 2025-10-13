static void update_reference_segmentation_map ( VP9_COMP * cpi ) {
     VP9_COMMON * const cm = & cpi -> common ;
     MODE_INFO * * mi_8x8_ptr = cm -> mi_grid_visible ;
     uint8_t * cache_ptr = cm -> last_frame_seg_map ;
     int row , col ;

     for ( row = 0 ; row < cm -> mi_rows ; row ++ ) {
         MODE_INFO * * mi_8x8 = mi_8x8_ptr ;
         uint8_t * cache = cache_ptr ;
         for ( col = 0 ; col < cm -> mi_cols ; col ++ , mi_8x8 ++ , cache ++ )
             cache [ 0 ] = mi_8x8 [ 0 ] -> mbmi . segment_id ;
         mi_8x8_ptr += cm -> mi_stride ;
         cache_ptr += cm -> mi_cols ;
         }
 }