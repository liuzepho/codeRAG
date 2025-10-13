int vp9_get_pred_context_single_ref_p2 ( const MACROBLOCKD * xd ) {
 int pred_context ;
 const MB_MODE_INFO * const above_mbmi = get_mbmi ( get_above_mi ( xd ) ) ;
 const MB_MODE_INFO * const left_mbmi = get_mbmi ( get_left_mi ( xd ) ) ;
 const int has_above = above_mbmi != NULL ;
 const int has_left = left_mbmi != NULL ;
 if ( has_above && has_left ) {
 const int above_intra = ! is_inter_block ( above_mbmi ) ;
 const int left_intra = ! is_inter_block ( left_mbmi ) ;
 if ( above_intra && left_intra ) {
 pred_context = 2 ;
 }
 else if ( above_intra || left_intra ) {
 const MB_MODE_INFO * edge_mbmi = above_intra ? left_mbmi : above_mbmi ;
 if ( ! has_second_ref ( edge_mbmi ) ) {
 if ( edge_mbmi -> ref_frame [ 0 ] == LAST_FRAME ) pred_context = 3 ;
 else pred_context = 4 * ( edge_mbmi -> ref_frame [ 0 ] == GOLDEN_FRAME ) ;
 }
 else {
 pred_context = 1 + 2 * ( edge_mbmi -> ref_frame [ 0 ] == GOLDEN_FRAME || edge_mbmi -> ref_frame [ 1 ] == GOLDEN_FRAME ) ;
 }
 }
 else {
 const int above_has_second = has_second_ref ( above_mbmi ) ;
 const int left_has_second = has_second_ref ( left_mbmi ) ;
 const MV_REFERENCE_FRAME above0 = above_mbmi -> ref_frame [ 0 ] ;
 const MV_REFERENCE_FRAME above1 = above_mbmi -> ref_frame [ 1 ] ;
 const MV_REFERENCE_FRAME left0 = left_mbmi -> ref_frame [ 0 ] ;
 const MV_REFERENCE_FRAME left1 = left_mbmi -> ref_frame [ 1 ] ;
 if ( above_has_second && left_has_second ) {
 if ( above0 == left0 && above1 == left1 ) pred_context = 3 * ( above0 == GOLDEN_FRAME || above1 == GOLDEN_FRAME || left0 == GOLDEN_FRAME || left1 == GOLDEN_FRAME ) ;
 else pred_context = 2 ;
 }
 else if ( above_has_second || left_has_second ) {
 const MV_REFERENCE_FRAME rfs = ! above_has_second ? above0 : left0 ;
 const MV_REFERENCE_FRAME crf1 = above_has_second ? above0 : left0 ;
 const MV_REFERENCE_FRAME crf2 = above_has_second ? above1 : left1 ;
 if ( rfs == GOLDEN_FRAME ) pred_context = 3 + ( crf1 == GOLDEN_FRAME || crf2 == GOLDEN_FRAME ) ;
 else if ( rfs == ALTREF_FRAME ) pred_context = crf1 == GOLDEN_FRAME || crf2 == GOLDEN_FRAME ;
 else pred_context = 1 + 2 * ( crf1 == GOLDEN_FRAME || crf2 == GOLDEN_FRAME ) ;
 }
 else {
 if ( above0 == LAST_FRAME && left0 == LAST_FRAME ) {
 pred_context = 3 ;
 }
 else if ( above0 == LAST_FRAME || left0 == LAST_FRAME ) {
 const MV_REFERENCE_FRAME edge0 = ( above0 == LAST_FRAME ) ? left0 : above0 ;
 pred_context = 4 * ( edge0 == GOLDEN_FRAME ) ;
 }
 else {
 pred_context = 2 * ( above0 == GOLDEN_FRAME ) + * ( left0 == GOLDEN_FRAME ) ;
 }
 }
 }
 }
 else if ( has_above || has_left ) {
 const MB_MODE_INFO * edge_mbmi = has_above ? above_mbmi : left_mbmi ;
 if ( ! is_inter_block ( edge_mbmi ) || ( edge_mbmi -> ref_frame [ 0 ] == LAST_FRAME && ! has_second_ref ( edge_mbmi ) ) ) pred_context = 2 ;
 else if ( ! has_second_ref ( edge_mbmi ) ) pred_context = 4 * ( edge_mbmi -> ref_frame [ 0 ] == GOLDEN_FRAME ) ;
 else pred_context = 3 * ( edge_mbmi -> ref_frame [ 0 ] == GOLDEN_FRAME || edge_mbmi -> ref_frame [ 1 ] == GOLDEN_FRAME ) ;
 }
 else {
 pred_context = 2 ;
 }
 assert ( pred_context >= 0 && pred_context < REF_CONTEXTS ) ;
 return pred_context ;
 }