	B?f??j??B?f??j??!B?f??j??	J̘i??!@J̘i??!@!J̘i??!@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$B?f??j????ZӼ???A^K?=???Y??ܵ??*	??????X@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??y?):??!2?c??A@)???{????1?9??sn@@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?&S???!c?1?XR@)c?ZB>???1"?B?9@:Preprocessing2F
Iterator::ModelF%u???!?s?9??:@)??_vO??1?c?1?5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate'???????!SJ)???%@)	?^)ˀ?1!?B? @:Preprocessing2U
Iterator::Model::ParallelMapV2a2U0*?s?![k???Z@)a2U0*?s?1[k???Z@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?~j?t?h?!2?c?1@)?~j?t?h?12?c?1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice{?G?zd?!)??RJ)@){?G?zd?1)??RJ)@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaptF??_??!???{??'@)a2U0*?S?1[k???Z??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 8.8% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t17.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9J̘i??!@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??ZӼ?????ZӼ???!??ZӼ???      ??!       "      ??!       *      ??!       2	^K?=???^K?=???!^K?=???:      ??!       B      ??!       J	??ܵ????ܵ??!??ܵ??R      ??!       Z	??ܵ????ܵ??!??ܵ??JCPU_ONLYYJ̘i??!@b 