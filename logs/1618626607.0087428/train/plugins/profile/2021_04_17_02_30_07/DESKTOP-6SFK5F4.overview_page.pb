?	??????????!?????	2!??(@2!??(@!2!??(@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?????aTR'????A?q?????Y؁sF????*	fffffFQ@2F
Iterator::Model????ׁ??!6???K@)?R?!?u??1^?}??H@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???Q???!??,?)?5@)9??v????1??HY5?2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?~j?t???!?W?]1@)??ׁsF??1?O?t?,@:Preprocessing2U
Iterator::Model::ParallelMapV2????Mbp?!???'@)????Mbp?1???'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipX9??v???!??%?fnF@)??_vOf?1F???PB@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?J?4a?!?y?P@)?J?4a?1?y?P@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Mb`?!???'@)????Mb`?1???'@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?]K?=??! )?Z?3@)??_?LU?1?A????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 12.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t17.3 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.93!??(@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	aTR'????aTR'????!aTR'????      ??!       "      ??!       *      ??!       2	?q??????q?????!?q?????:      ??!       B      ??!       J	؁sF????؁sF????!؁sF????R      ??!       Z	؁sF????؁sF????!؁sF????JCPU_ONLYY3!??(@b Y      Y@qA(V?P/P@"?	
both?Your program is MODERATELY input-bound because 12.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nohigh"t17.3 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.:
Refer to the TF2 Profiler FAQb?64.7393% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 