# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: task.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\ntask.proto\x12\x0b\x61lphaevolve\"6\n\x0eTaskCollection\x12$\n\x05tasks\x18\x01 \x03(\x0b\x32\x15.alphaevolve.TaskSpec\"\xf5\x05\n\x08TaskSpec\x12\x15\n\rfeatures_size\x18\r \x01(\x05\x12\x1a\n\x12num_train_examples\x18\x01 \x01(\x05\x12\x1b\n\x10num_train_epochs\x18\x15 \x01(\x05:\x01\x31\x12\x1a\n\x12num_valid_examples\x18\x02 \x01(\x05\x12\x11\n\tnum_tasks\x18\x03 \x01(\x05\x12\x12\n\ndata_seeds\x18\x04 \x03(\r\x12\x13\n\x0bparam_seeds\x18\x05 \x03(\r\x12(\n\teval_type\x18\x1c \x01(\x0e\x32\x15.alphaevolve.EvalType\x12T\n\x1dscalar_linear_regression_task\x18\x06 \x01(\x0b\x32+.alphaevolve.ScalarLinearRegressionTaskSpecH\x00\x12Y\n scalar_2layer_nn_regression_task\x18\x07 \x01(\x0b\x32-.alphaevolve.Scalar2LayerNNRegressionTaskSpecH\x00\x12,\n\nstock_task\x18\x18 \x01(\x0b\x32\x16.alphaevolve.StockTaskH\x00\x12>\n\x14unit_test_fixed_task\x18( \x01(\x0b\x32\x1e.alphaevolve.UnitTestFixedTaskH\x00\x12\x42\n\x14unit_test_zeros_task\x18- \x01(\x0b\x32\".alphaevolve.UnitTestZerosTaskSpecH\x00\x12@\n\x13unit_test_ones_task\x18. \x01(\x0b\x32!.alphaevolve.UnitTestOnesTaskSpecH\x00\x12J\n\x18unit_test_increment_task\x18/ \x01(\x0b\x32&.alphaevolve.UnitTestIncrementTaskSpecH\x00\x12\x19\n\x11num_test_examples\x18\x12 \x01(\x05\x42\x0b\n\ttask_type\" \n\x1eScalarLinearRegressionTaskSpec\"\"\n Scalar2LayerNNRegressionTaskSpec\"\x9d\x02\n\tStockTask\x12\x16\n\x0epositive_class\x18\x01 \x01(\x05\x12\x16\n\x0enegative_class\x18\x02 \x01(\x05\x12\x14\n\x0c\x64\x61taset_name\x18\x03 \x01(\t\x12\x0e\n\x04path\x18\x04 \x01(\tH\x00\x12\x32\n\x07\x64\x61taset\x18\x05 \x01(\x0b\x32\x1f.alphaevolve.ScalarLabelDatasetH\x00\x12.\n\x0eheld_out_pairs\x18\x06 \x03(\x0b\x32\x16.alphaevolve.ClassPair\x12\"\n\x17min_supported_data_seed\x18\x07 \x01(\x05:\x01\x30\x12#\n\x17max_supported_data_seed\x18\x08 \x01(\x05:\x02\x31\x30\x42\r\n\x0btask_source\";\n\tClassPair\x12\x16\n\x0epositive_class\x18\x01 \x01(\x05\x12\x16\n\x0enegative_class\x18\x02 \x01(\x05\"\xf0\x01\n\x12ScalarLabelDataset\x12\x32\n\x0etrain_features\x18\x01 \x03(\x0b\x32\x1a.alphaevolve.FeatureVector\x12\x14\n\x0ctrain_labels\x18\x02 \x03(\x02\x12\x32\n\x0evalid_features\x18\x03 \x03(\x0b\x32\x1a.alphaevolve.FeatureVector\x12\x14\n\x0cvalid_labels\x18\x04 \x03(\x02\x12\x31\n\rtest_features\x18\x05 \x03(\x0b\x32\x1a.alphaevolve.FeatureVector\x12\x13\n\x0btest_labels\x18\x06 \x03(\x02\"!\n\rFeatureVector\x12\x10\n\x08\x66\x65\x61tures\x18\x01 \x03(\x02\"\x85\x03\n\x11UnitTestFixedTask\x12<\n\x0etrain_features\x18\x01 \x03(\x0b\x32$.alphaevolve.UnitTestFixedTaskVector\x12:\n\x0ctrain_labels\x18\x02 \x03(\x0b\x32$.alphaevolve.UnitTestFixedTaskVector\x12<\n\x0evalid_features\x18\x03 \x03(\x0b\x32$.alphaevolve.UnitTestFixedTaskVector\x12:\n\x0cvalid_labels\x18\x04 \x03(\x0b\x32$.alphaevolve.UnitTestFixedTaskVector\x12;\n\rtest_features\x18\x05 \x03(\x0b\x32$.alphaevolve.UnitTestFixedTaskVector\x12\x39\n\x0btest_labels\x18\x06 \x03(\x0b\x32$.alphaevolve.UnitTestFixedTaskVectorJ\x04\x08\x07\x10\x08\"+\n\x17UnitTestFixedTaskVector\x12\x10\n\x08\x65lements\x18\x01 \x03(\x01\"\x17\n\x15UnitTestZerosTaskSpec\"\x16\n\x14UnitTestOnesTaskSpec\"1\n\x19UnitTestIncrementTaskSpec\x12\x14\n\tincrement\x18\x01 \x01(\x01:\x01\x31*>\n\x08\x45valType\x12\x15\n\x11INVALID_EVAL_TYPE\x10\x00\x12\r\n\tRMS_ERROR\x10\x01\x12\x0c\n\x08\x41\x43\x43URACY\x10\x04*$\n\x0e\x41\x63tivationType\x12\x08\n\x04RELU\x10\x00\x12\x08\n\x04TANH\x10\x01')

_EVALTYPE = DESCRIPTOR.enum_types_by_name['EvalType']
EvalType = enum_type_wrapper.EnumTypeWrapper(_EVALTYPE)
_ACTIVATIONTYPE = DESCRIPTOR.enum_types_by_name['ActivationType']
ActivationType = enum_type_wrapper.EnumTypeWrapper(_ACTIVATIONTYPE)
INVALID_EVAL_TYPE = 0
RMS_ERROR = 1
ACCURACY = 4
RELU = 0
TANH = 1


_TASKCOLLECTION = DESCRIPTOR.message_types_by_name['TaskCollection']
_TASKSPEC = DESCRIPTOR.message_types_by_name['TaskSpec']
_SCALARLINEARREGRESSIONTASKSPEC = DESCRIPTOR.message_types_by_name['ScalarLinearRegressionTaskSpec']
_SCALAR2LAYERNNREGRESSIONTASKSPEC = DESCRIPTOR.message_types_by_name['Scalar2LayerNNRegressionTaskSpec']
_STOCKTASK = DESCRIPTOR.message_types_by_name['StockTask']
_CLASSPAIR = DESCRIPTOR.message_types_by_name['ClassPair']
_SCALARLABELDATASET = DESCRIPTOR.message_types_by_name['ScalarLabelDataset']
_FEATUREVECTOR = DESCRIPTOR.message_types_by_name['FeatureVector']
_UNITTESTFIXEDTASK = DESCRIPTOR.message_types_by_name['UnitTestFixedTask']
_UNITTESTFIXEDTASKVECTOR = DESCRIPTOR.message_types_by_name['UnitTestFixedTaskVector']
_UNITTESTZEROSTASKSPEC = DESCRIPTOR.message_types_by_name['UnitTestZerosTaskSpec']
_UNITTESTONESTASKSPEC = DESCRIPTOR.message_types_by_name['UnitTestOnesTaskSpec']
_UNITTESTINCREMENTTASKSPEC = DESCRIPTOR.message_types_by_name['UnitTestIncrementTaskSpec']
TaskCollection = _reflection.GeneratedProtocolMessageType('TaskCollection', (_message.Message,), {
  'DESCRIPTOR' : _TASKCOLLECTION,
  '__module__' : 'task_pb2'
  # @@protoc_insertion_point(class_scope:alphaevolve.TaskCollection)
  })
_sym_db.RegisterMessage(TaskCollection)

TaskSpec = _reflection.GeneratedProtocolMessageType('TaskSpec', (_message.Message,), {
  'DESCRIPTOR' : _TASKSPEC,
  '__module__' : 'task_pb2'
  # @@protoc_insertion_point(class_scope:alphaevolve.TaskSpec)
  })
_sym_db.RegisterMessage(TaskSpec)

ScalarLinearRegressionTaskSpec = _reflection.GeneratedProtocolMessageType('ScalarLinearRegressionTaskSpec', (_message.Message,), {
  'DESCRIPTOR' : _SCALARLINEARREGRESSIONTASKSPEC,
  '__module__' : 'task_pb2'
  # @@protoc_insertion_point(class_scope:alphaevolve.ScalarLinearRegressionTaskSpec)
  })
_sym_db.RegisterMessage(ScalarLinearRegressionTaskSpec)

Scalar2LayerNNRegressionTaskSpec = _reflection.GeneratedProtocolMessageType('Scalar2LayerNNRegressionTaskSpec', (_message.Message,), {
  'DESCRIPTOR' : _SCALAR2LAYERNNREGRESSIONTASKSPEC,
  '__module__' : 'task_pb2'
  # @@protoc_insertion_point(class_scope:alphaevolve.Scalar2LayerNNRegressionTaskSpec)
  })
_sym_db.RegisterMessage(Scalar2LayerNNRegressionTaskSpec)

StockTask = _reflection.GeneratedProtocolMessageType('StockTask', (_message.Message,), {
  'DESCRIPTOR' : _STOCKTASK,
  '__module__' : 'task_pb2'
  # @@protoc_insertion_point(class_scope:alphaevolve.StockTask)
  })
_sym_db.RegisterMessage(StockTask)

ClassPair = _reflection.GeneratedProtocolMessageType('ClassPair', (_message.Message,), {
  'DESCRIPTOR' : _CLASSPAIR,
  '__module__' : 'task_pb2'
  # @@protoc_insertion_point(class_scope:alphaevolve.ClassPair)
  })
_sym_db.RegisterMessage(ClassPair)

ScalarLabelDataset = _reflection.GeneratedProtocolMessageType('ScalarLabelDataset', (_message.Message,), {
  'DESCRIPTOR' : _SCALARLABELDATASET,
  '__module__' : 'task_pb2'
  # @@protoc_insertion_point(class_scope:alphaevolve.ScalarLabelDataset)
  })
_sym_db.RegisterMessage(ScalarLabelDataset)

FeatureVector = _reflection.GeneratedProtocolMessageType('FeatureVector', (_message.Message,), {
  'DESCRIPTOR' : _FEATUREVECTOR,
  '__module__' : 'task_pb2'
  # @@protoc_insertion_point(class_scope:alphaevolve.FeatureVector)
  })
_sym_db.RegisterMessage(FeatureVector)

UnitTestFixedTask = _reflection.GeneratedProtocolMessageType('UnitTestFixedTask', (_message.Message,), {
  'DESCRIPTOR' : _UNITTESTFIXEDTASK,
  '__module__' : 'task_pb2'
  # @@protoc_insertion_point(class_scope:alphaevolve.UnitTestFixedTask)
  })
_sym_db.RegisterMessage(UnitTestFixedTask)

UnitTestFixedTaskVector = _reflection.GeneratedProtocolMessageType('UnitTestFixedTaskVector', (_message.Message,), {
  'DESCRIPTOR' : _UNITTESTFIXEDTASKVECTOR,
  '__module__' : 'task_pb2'
  # @@protoc_insertion_point(class_scope:alphaevolve.UnitTestFixedTaskVector)
  })
_sym_db.RegisterMessage(UnitTestFixedTaskVector)

UnitTestZerosTaskSpec = _reflection.GeneratedProtocolMessageType('UnitTestZerosTaskSpec', (_message.Message,), {
  'DESCRIPTOR' : _UNITTESTZEROSTASKSPEC,
  '__module__' : 'task_pb2'
  # @@protoc_insertion_point(class_scope:alphaevolve.UnitTestZerosTaskSpec)
  })
_sym_db.RegisterMessage(UnitTestZerosTaskSpec)

UnitTestOnesTaskSpec = _reflection.GeneratedProtocolMessageType('UnitTestOnesTaskSpec', (_message.Message,), {
  'DESCRIPTOR' : _UNITTESTONESTASKSPEC,
  '__module__' : 'task_pb2'
  # @@protoc_insertion_point(class_scope:alphaevolve.UnitTestOnesTaskSpec)
  })
_sym_db.RegisterMessage(UnitTestOnesTaskSpec)

UnitTestIncrementTaskSpec = _reflection.GeneratedProtocolMessageType('UnitTestIncrementTaskSpec', (_message.Message,), {
  'DESCRIPTOR' : _UNITTESTINCREMENTTASKSPEC,
  '__module__' : 'task_pb2'
  # @@protoc_insertion_point(class_scope:alphaevolve.UnitTestIncrementTaskSpec)
  })
_sym_db.RegisterMessage(UnitTestIncrementTaskSpec)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _EVALTYPE._serialized_start=2077
  _EVALTYPE._serialized_end=2139
  _ACTIVATIONTYPE._serialized_start=2141
  _ACTIVATIONTYPE._serialized_end=2177
  _TASKCOLLECTION._serialized_start=27
  _TASKCOLLECTION._serialized_end=81
  _TASKSPEC._serialized_start=84
  _TASKSPEC._serialized_end=841
  _SCALARLINEARREGRESSIONTASKSPEC._serialized_start=843
  _SCALARLINEARREGRESSIONTASKSPEC._serialized_end=875
  _SCALAR2LAYERNNREGRESSIONTASKSPEC._serialized_start=877
  _SCALAR2LAYERNNREGRESSIONTASKSPEC._serialized_end=911
  _STOCKTASK._serialized_start=914
  _STOCKTASK._serialized_end=1199
  _CLASSPAIR._serialized_start=1201
  _CLASSPAIR._serialized_end=1260
  _SCALARLABELDATASET._serialized_start=1263
  _SCALARLABELDATASET._serialized_end=1503
  _FEATUREVECTOR._serialized_start=1505
  _FEATUREVECTOR._serialized_end=1538
  _UNITTESTFIXEDTASK._serialized_start=1541
  _UNITTESTFIXEDTASK._serialized_end=1930
  _UNITTESTFIXEDTASKVECTOR._serialized_start=1932
  _UNITTESTFIXEDTASKVECTOR._serialized_end=1975
  _UNITTESTZEROSTASKSPEC._serialized_start=1977
  _UNITTESTZEROSTASKSPEC._serialized_end=2000
  _UNITTESTONESTASKSPEC._serialized_start=2002
  _UNITTESTONESTASKSPEC._serialized_end=2024
  _UNITTESTINCREMENTTASKSPEC._serialized_start=2026
  _UNITTESTINCREMENTTASKSPEC._serialized_end=2075
# @@protoc_insertion_point(module_scope)