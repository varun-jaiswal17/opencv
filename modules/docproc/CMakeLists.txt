set(the_description "Document reading convenience API built on top of the vlm module")

ocv_add_module(docproc opencv_core opencv_imgcodecs opencv_vlm WRAP python)
ocv_glob_module_sources()
ocv_module_include_directories()
ocv_create_module()

ocv_add_accuracy_tests()
ocv_add_perf_tests()
ocv_add_samples()
