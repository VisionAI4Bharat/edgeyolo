# Graph Report - deepSightAI  (2026-04-22)

## Corpus Check
- 775 files · ~999,866 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 3145 nodes · 5584 edges · 48 communities detected
- Extraction: 75% EXTRACTED · 25% INFERRED · 0% AMBIGUOUS · INFERRED: 1369 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 73|Community 73]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 77|Community 77]]
- [[_COMMUNITY_Community 78|Community 78]]
- [[_COMMUNITY_Community 132|Community 132]]
- [[_COMMUNITY_Community 134|Community 134]]
- [[_COMMUNITY_Community 136|Community 136]]

## God Nodes (most connected - your core abstractions)
1. `v_reinterpret_as_u8()` - 104 edges
2. `v_reinterpret_as_u16()` - 102 edges
3. `v_reinterpret_as_u32()` - 99 edges
4. `v_reinterpret_as_s8()` - 67 edges
5. `v_reinterpret_as_s16()` - 62 edges
6. `v_reinterpret_as_u64()` - 62 edges
7. `size()` - 59 edges
8. `v_reinterpret_as_s32()` - 54 edges
9. `empty()` - 36 edges
10. `execute()` - 33 edges

## Surprising Connections (you probably didn't know these)
- `internal_add_copy()` --calls--> `type()`  [INFERRED]
  temp/Linux_x86_64/tbb/include/oneapi/tbb/parallel_for_each.h → third_party/rockchip/RV110X/mpi/include/opencv4/opencv2/core/mat.inl.hpp
- `run()` --calls--> `empty()`  [INFERRED]
  temp/Linux_x86_64/tbb/include/oneapi/tbb/parallel_reduce.h → third_party/rockchip/RV110X/mpi/include/opencv4/opencv2/core/types.hpp
- `cpq_operation()` --calls--> `type()`  [INFERRED]
  temp/Linux_x86_64/tbb/include/oneapi/tbb/concurrent_priority_queue.h → third_party/rockchip/RV110X/mpi/include/opencv4/opencv2/core/mat.inl.hpp
- `destruct_and_deallocate()` --calls--> `deallocate()`  [INFERRED]
  temp/Linux_x86_64/tbb/include/oneapi/tbb/detail/_flow_graph_impl.h → third_party/rockchip/RV110X/mpi/include/opencv4/opencv2/core/utility.hpp
- `const_range_type()` --calls--> `next()`  [INFERRED]
  temp/Linux_x86_64/tbb/include/oneapi/tbb/detail/_concurrent_skip_list.h → third_party/rockchip/RV110X/mpi/include/opencv4/opencv2/core/operations.hpp

## Communities

### Community 0 - "Community 0"
Cohesion: 0.02
Nodes (152): dsai_fps(), dsai_height(), dsai_loadFromFile(), dsai_saveToFile(), dsai_width(), generateVectorOfInt(), generateVectorOfRect(), handle_operations() (+144 more)

### Community 1 - "Community 1"
Cohesion: 0.04
Nodes (191): _v512_combine(), _v512_extract_high(), _v512_extract_low(), v512_load_expand(), v512_lut(), v512_lut_pairs(), v512_lut_quads(), v_abs() (+183 more)

### Community 2 - "Community 2"
Cohesion: 0.03
Nodes (130): a(), advance_to_next_bucket(), allocate_node_copy_construct(), allocate_node_move_construct(), begin(), bucket_accessor(), check_mask_race(), check_rehashing_collision() (+122 more)

### Community 3 - "Community 3"
Cohesion: 0.02
Nodes (101): try_pop(), advance(), assign(), clear(), clear_and_invalidate(), get_item(), my_item(), tbb() (+93 more)

### Community 4 - "Community 4"
Cohesion: 0.02
Nodes (90): Affine3, Affine3<T>::Affine3(), Affine3<T>::inv(), Affine3<T>::rotate(), Affine3<T>::rotation(), DataType< Affine3<_Tp> >, Transform<T, 3, Eigen::Affine, (Eigen::RowMajor)>(), Mat_ (+82 more)

### Community 5 - "Community 5"
Cohesion: 0.04
Nodes (96): Affine3<T>::rvec(), Complex, cv_abs(), CV_EXPORTS, MatConstIterator_, MatIterator_, Matx, normInf() (+88 more)

### Community 6 - "Community 6"
Cohesion: 0.04
Nodes (79): v_invsqrt(), _lsx_128_castpd_si128(), _lsx_128_castps_si128(), _lsx_packs_h(), _lsx_packus_h(), _v128_set_d(), _v128_setr_b(), _v128_setr_d() (+71 more)

### Community 7 - "Community 7"
Cohesion: 0.05
Nodes (73): _lasx_256_castpd_si256(), _lasx_256_castps_si256(), _lasx_packs_h(), _lasx_packs_w(), _lasx_packus_h(), _lasx_packus_w(), v256_alignr_64(), _v256_alignr_b() (+65 more)

### Community 8 - "Community 8"
Cohesion: 0.06
Nodes (80): v_absdiffs(), v_absdiffs(), v_absdiffs(), v_absdiffs(), v_abs(), v_absdiff(), v_absdiffs(), v_broadcast_element() (+72 more)

### Community 9 - "Community 9"
Cohesion: 0.05
Nodes (68): v_transpose4x4(), v_store_interleave(), v128_cvti16x8_i32x4(), v128_cvti16x8_i32x4_high(), v128_cvti32x4_i64x2(), v128_cvti32x4_i64x2_high(), v128_cvti8x16_i16x8(), v128_cvti8x16_i16x8_high() (+60 more)

### Community 10 - "Community 10"
Cohesion: 0.05
Nodes (62): v256_alignr_64(), v256_blend(), _v256_combine(), _v256_extract_high(), _v256_extract_low(), v256_load_expand(), _v256_packs_epu32(), _v256_shuffle_odd_64() (+54 more)

### Community 11 - "Community 11"
Cohesion: 0.05
Nodes (54): dsai_defaultPath(), ClassLabelsDialog(), dsai_addLabel(), dsai_deleteLabel(), dsai_getClassLabels(), dsai_moveDown(), dsai_moveUp(), dsai_readFromYaml() (+46 more)

### Community 12 - "Community 12"
Cohesion: 0.04
Nodes (33): decltype(), Ort(), OrtEnv(), OrtSessionOptions(), OrtThreadingOptions(), CreateAndRegisterAllocator(), CreateCpu(), detail() (+25 more)

### Community 13 - "Community 13"
Cohesion: 0.05
Nodes (44): erase(), assign(), cpq_operation(), heapify(), push(), push_back_helper(), push_back_helper_impl(), tbb() (+36 more)

### Community 14 - "Community 14"
Cohesion: 0.05
Nodes (37): create(), dumpCString(), dumpDouble(), dumpFloat(), dumpInt(), dumpRange(), dumpRect(), dumpRotatedRect() (+29 more)

### Community 15 - "Community 15"
Cohesion: 0.07
Nodes (37): v_abs(), v_ceil(), v_cvt_f32(), v_cvt_f64(), v_cvt_f64_high(), v_dotprod(), v_dotprod_expand(), v_dotprod_expand_fast() (+29 more)

### Community 16 - "Community 16"
Cohesion: 0.1
Nodes (41): v_abs(), v_ceil(), v_cvt_f32(), v_cvt_f64(), v_cvt_f64_high(), v_div(), v_dotprod(), v_dotprod_expand() (+33 more)

### Community 17 - "Community 17"
Cohesion: 0.12
Nodes (42): barrier1(), __builtin_riscv_fsrm(), OPENCV_HAL_IMPL_RISCVV_EXPAND(), OPENCV_HAL_IMPL_RISCVV_INIT_SET(), OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(), v_abs(), v_absdiffs(), v_ceil() (+34 more)

### Community 18 - "Community 18"
Cohesion: 0.08
Nodes (34): v_ceil(), v_cvt_f32(), v_cvt_f64(), v_cvt_f64_high(), v_dotprod(), v_dotprod_expand(), v_dotprod_expand_fast(), v_dotprod_fast() (+26 more)

### Community 19 - "Community 19"
Cohesion: 0.06
Nodes (19): vmul(), vreinterpret_u32mf2(), v_cvt_f64(), v_dotprod(), v_dotprod_expand(), v_dotprod_expand_fast(), v_dotprod_fast(), v_fma() (+11 more)

### Community 20 - "Community 20"
Cohesion: 0.06
Nodes (17): all(), DataType< Matx<_Tp, m, n> >, DataType< Vec<_Tp, cn> >, div(), Matx(), MatxCommaInitializer, mul(), norm() (+9 more)

### Community 22 - "Community 22"
Cohesion: 0.06
Nodes (6): BOWKMeansTrainer(), CV_EXPORTS_W, MinProblemSolver(), KalmanFilter(), SparsePyrLKOpticalFlow(), TermCriteria()

### Community 23 - "Community 23"
Cohesion: 0.09
Nodes (15): AutoBuffer, findFileOrKeep(), forEach_impl(), getAvgTimeMilli(), getAvgTimeSec(), getFPS(), getTimeMicro(), getTimeMilli() (+7 more)

### Community 24 - "Community 24"
Cohesion: 0.09
Nodes (6): cvReadInt(), cvReadIntByName(), cvReadReal(), cvReadRealByName(), cvReadString(), cvReadStringByName()

### Community 25 - "Community 25"
Cohesion: 0.09
Nodes (3): fgt_body(), fgt_multioutput_node_with_body(), fgt_node_with_body()

### Community 26 - "Community 26"
Cohesion: 0.24
Nodes (23): buffer_full(), capacity(), clean_up_buffer(), consume_front(), destroy_back(), destroy_front(), destroy_item(), fetch_item() (+15 more)

### Community 27 - "Community 27"
Cohesion: 0.11
Nodes (12): CV_EXPORTS, SparseMatIterator(), _InputArray(), Mat(), MatCommaInitializer_, MatExpr(), MatIterator_, max() (+4 more)

### Community 28 - "Community 28"
Cohesion: 0.09
Nodes (21): DataDepth, DataType, DataType<bool>, DataType<char>, DataType<double>, DataType<float>, DataType<hfloat>, DataType<int> (+13 more)

### Community 29 - "Community 29"
Cohesion: 0.11
Nodes (4): _v128_blendv_epi8(), _v128_comgt_epu32(), _v128_min_epu32(), _v128_packs_epu32()

### Community 30 - "Community 30"
Cohesion: 0.17
Nodes (13): clear_list(), INIT_LIST_HEAD(), __list_add(), list_add_tail(), __list_del(), list_del_init(), list_empty(), list_move() (+5 more)

### Community 31 - "Community 31"
Cohesion: 0.24
Nodes (4): isolated_task_group(), run(), task_group_base(), tbb()

### Community 33 - "Community 33"
Cohesion: 0.18
Nodes (1): CV_EXPORTS

### Community 34 - "Community 34"
Cohesion: 0.25
Nodes (6): cv_vrecp_f32(), cv_vrecpq_f32(), cv_vrsqrt_f32(), cv_vrsqrtq_f32(), cv_vsqrt_f32(), cv_vsqrtq_f32()

### Community 35 - "Community 35"
Cohesion: 0.27
Nodes (4): debug_wait_until_empty(), enqueue(), initialize(), tbb()

### Community 36 - "Community 36"
Cohesion: 0.25
Nodes (2): AreZero(), onnxruntime_float16()

### Community 37 - "Community 37"
Cohesion: 0.22
Nodes (2): calibdb_get_module_ptr(), calibdbV2_get_module_ptr()

### Community 42 - "Community 42"
Cohesion: 0.5
Nodes (2): cvRandInt(), cvRandReal()

### Community 43 - "Community 43"
Cohesion: 0.4
Nodes (2): FPDenormalsIgnoreHintScope, FPDenormalsIgnoreHintScopeNOOP

### Community 44 - "Community 44"
Cohesion: 0.4
Nodes (2): CallbackProxy, ParallelForBackend

### Community 56 - "Community 56"
Cohesion: 0.67
Nodes (1): tbb()

### Community 73 - "Community 73"
Cohesion: 0.67
Nodes (2): Quat, QuatEnum

### Community 74 - "Community 74"
Cohesion: 0.67
Nodes (1): BufferPoolController

### Community 76 - "Community 76"
Cohesion: 0.67
Nodes (1): AllocatorStatisticsInterface

### Community 77 - "Community 77"
Cohesion: 0.67
Nodes (1): ParallelForBackend

### Community 78 - "Community 78"
Cohesion: 0.67
Nodes (1): VideoCaptureImpl

### Community 132 - "Community 132"
Cohesion: 1.0
Nodes (1): DualQuat

### Community 134 - "Community 134"
Cohesion: 1.0
Nodes (1): DataType< std::complex<_Tp> >

### Community 136 - "Community 136"
Cohesion: 1.0
Nodes (1): AllocatorStatistics

## Knowledge Gaps
- **95 isolated node(s):** `CV_EXPORTS`, `FontFaceImpl`, `CV_EXPORTS_W`, `CV_EXPORTS`, `Vec` (+90 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 33`** (11 nodes): `Algorithm()`, `CV_EXPORTS`, `exception()`, `Formatted()`, `Formatter()`, `LDA()`, `PCA()`, `RNG()`, `RNG_MT19937()`, `SVD()`, `core.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (9 nodes): `Abs()`, `AreZero()`, `detail()`, `fl()`, `IsNormal()`, `onnxruntime_float16()`, `ToFloatImpl()`, `val()`, `onnxruntime_float16.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (9 nodes): `calibdb_get_module_ptr()`, `calibdbV2_from_tuningdb()`, `calibdbV2_get_module_ptr()`, `calibdbv2_get_scene_ctx_struct_name()`, `calibdbv2_get_scene_ptr()`, `calibdbV2_scene_ctx_size()`, `calibdbV2_to_tuningdb()`, `RkAiqCalibDbTypes.h`, `RkAiqCalibDbV2Helper.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (5 nodes): `types_c.h`, `cvIplImage()`, `cvRandInt()`, `cvRandReal()`, `cvRNG()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (5 nodes): `FPDenormalsIgnoreHintScope`, `.FPDenormalsIgnoreHintScope()`, `FPDenormalsIgnoreHintScopeNOOP`, `.FPDenormalsIgnoreHintScopeNOOP()`, `fp_control_utils.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (5 nodes): `CallbackProxy`, `.CallbackProxy()`, `ParallelForBackend`, `.ParallelForBackend()`, `parallel_for.tbb.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (3 nodes): `tbb()`, `_task.h`, `task.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 73`** (3 nodes): `Quat`, `QuatEnum`, `quaternion.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 74`** (3 nodes): `BufferPoolController`, `.BufferPoolController()`, `bufferpool.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 76`** (3 nodes): `AllocatorStatisticsInterface`, `.AllocatorStatisticsInterface()`, `allocator_stats.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 77`** (3 nodes): `ParallelForBackend`, `.ParallelForBackend()`, `parallel_for.openmp.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 78`** (3 nodes): `VideoCapture()`, `VideoCaptureImpl`, `highgui.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 132`** (2 nodes): `DualQuat`, `dualquaternion.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 134`** (2 nodes): `DataType< std::complex<_Tp> >`, `cvstd.inl.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 136`** (2 nodes): `AllocatorStatistics`, `allocator_stats.impl.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `cvRound()` connect `Community 5` to `Community 24`, `Community 9`?**
  _High betweenness centrality (0.247) - this node is a cross-community bridge._
- **Why does `v_round()` connect `Community 9` to `Community 5`?**
  _High betweenness centrality (0.236) - this node is a cross-community bridge._
- **Why does `min()` connect `Community 5` to `Community 0`, `Community 2`, `Community 12`?**
  _High betweenness centrality (0.157) - this node is a cross-community bridge._
- **Are the 97 inferred relationships involving `v_reinterpret_as_u8()` (e.g. with `v_absdiff()` and `v_reverse()`) actually correct?**
  _`v_reinterpret_as_u8()` has 97 INFERRED edges - model-reasoned connections that need verification._
- **Are the 95 inferred relationships involving `v_reinterpret_as_u16()` (e.g. with `v_absdiff()` and `v_reverse()`) actually correct?**
  _`v_reinterpret_as_u16()` has 95 INFERRED edges - model-reasoned connections that need verification._
- **Are the 90 inferred relationships involving `v_reinterpret_as_u32()` (e.g. with `v_absdiff()` and `v_reverse()`) actually correct?**
  _`v_reinterpret_as_u32()` has 90 INFERRED edges - model-reasoned connections that need verification._
- **Are the 65 inferred relationships involving `v_reinterpret_as_s8()` (e.g. with `v_reverse()` and `v_scan_forward()`) actually correct?**
  _`v_reinterpret_as_s8()` has 65 INFERRED edges - model-reasoned connections that need verification._