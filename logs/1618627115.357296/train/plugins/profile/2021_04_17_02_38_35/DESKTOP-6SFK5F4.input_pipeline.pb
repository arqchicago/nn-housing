	?B?i?q???B?i?q??!?B?i?q??	Zk?k`B@Zk?k`B@!Zk?k`B@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?B?i?q??H?}8g??A?z?G???YԚ?????*	????̔u@2F
Iterator::Modelj?t???!s\u?ݻV@)O@a????1zz|nV@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?ZӼ???!˿?s@)??_vO??1?}6u?	@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?(??0??!?+E?8@)??_?L??1
?vAQ@:Preprocessing2U
Iterator::Model::ParallelMapV2?J?4q?!͚?>v??)?J?4q?1͚?>v??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipǺ????!eU?!"@)ŏ1w-!o?1O9ם???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice_?Q?k?!?t?B???)_?Q?k?1?t?B???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!O9ם???)ŏ1w-!_?1O9ם???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapŏ1w-!??!O9ם?@)????MbP?1???
܈??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 36.8% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2s9.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9[k?k`B@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	H?}8g??H?}8g??!H?}8g??      ??!       "      ??!       *      ??!       2	?z?G????z?G???!?z?G???:      ??!       B      ??!       J	Ԛ?????Ԛ?????!Ԛ?????R      ??!       Z	Ԛ?????Ԛ?????!Ԛ?????JCPU_ONLYY[k?k`B@b 