Ļµ
ßĆ
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
„
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12unknown8

embedding_88/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	½{U*(
shared_nameembedding_88/embeddings

+embedding_88/embeddings/Read/ReadVariableOpReadVariableOpembedding_88/embeddings*
_output_shapes
:	½{U*
dtype0
{
dense_78/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	!* 
shared_namedense_78/kernel
t
#dense_78/kernel/Read/ReadVariableOpReadVariableOpdense_78/kernel*
_output_shapes
:	!*
dtype0
r
dense_78/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_78/bias
k
!dense_78/bias/Read/ReadVariableOpReadVariableOpdense_78/bias*
_output_shapes
:*
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
t
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nametrue_positives
m
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes
:*
dtype0
v
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_positives
o
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes
:*
dtype0
x
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_1
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:*
dtype0
v
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_negatives
o
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes
:*
dtype0
£
#RMSprop/embedding_88/embeddings/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	½{U*4
shared_name%#RMSprop/embedding_88/embeddings/rms

7RMSprop/embedding_88/embeddings/rms/Read/ReadVariableOpReadVariableOp#RMSprop/embedding_88/embeddings/rms*
_output_shapes
:	½{U*
dtype0

RMSprop/dense_78/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	!*,
shared_nameRMSprop/dense_78/kernel/rms

/RMSprop/dense_78/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_78/kernel/rms*
_output_shapes
:	!*
dtype0

RMSprop/dense_78/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameRMSprop/dense_78/bias/rms

-RMSprop/dense_78/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_78/bias/rms*
_output_shapes
:*
dtype0

NoOpNoOp
É 
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0* 
valueśB÷ Bš
Ł
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
	optimizer
trainable_variables
regularization_losses
	variables
		keras_api


signatures
b

embeddings
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
`
iter
	decay
 learning_rate
!momentum
"rho	rmsS	rmsT	rmsU

0
1
2
 

0
1
2
­
#metrics
trainable_variables
$layer_regularization_losses
regularization_losses
	variables
%non_trainable_variables
&layer_metrics

'layers
 
ge
VARIABLE_VALUEembedding_88/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
­
(metrics
trainable_variables
)layer_regularization_losses
regularization_losses
	variables
*non_trainable_variables
+layer_metrics

,layers
 
 
 
­
-metrics
trainable_variables
.layer_regularization_losses
regularization_losses
	variables
/non_trainable_variables
0layer_metrics

1layers
 
 
 
­
2metrics
trainable_variables
3layer_regularization_losses
regularization_losses
	variables
4non_trainable_variables
5layer_metrics

6layers
[Y
VARIABLE_VALUEdense_78/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_78/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
7metrics
trainable_variables
8layer_regularization_losses
regularization_losses
	variables
9non_trainable_variables
:layer_metrics

;layers
KI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE

<0
=1
>2
?3
 
 
 

0
1
2
3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	@total
	Acount
B	variables
C	keras_api
D
	Dtotal
	Ecount
F
_fn_kwargs
G	variables
H	keras_api
W
I
thresholds
Jtrue_positives
Kfalse_positives
L	variables
M	keras_api
W
N
thresholds
Otrue_positives
Pfalse_negatives
Q	variables
R	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

@0
A1

B	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

D0
E1

G	variables
 
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE

J0
K1

L	variables
 
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

O0
P1

Q	variables

VARIABLE_VALUE#RMSprop/embedding_88/embeddings/rmsXlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/dense_78/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/dense_78/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

"serving_default_embedding_88_inputPlaceholder*'
_output_shapes
:’’’’’’’’’2*
dtype0*
shape:’’’’’’’’’2
ü
StatefulPartitionedCallStatefulPartitionedCall"serving_default_embedding_88_inputembedding_88/embeddingsdense_78/kerneldense_78/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_873498
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
÷
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+embedding_88/embeddings/Read/ReadVariableOp#dense_78/kernel/Read/ReadVariableOp!dense_78/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp7RMSprop/embedding_88/embeddings/rms/Read/ReadVariableOp/RMSprop/dense_78/kernel/rms/Read/ReadVariableOp-RMSprop/dense_78/bias/rms/Read/ReadVariableOpConst* 
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_873722
ö
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_88/embeddingsdense_78/kerneldense_78/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototalcounttotal_1count_1true_positivesfalse_positivestrue_positives_1false_negatives#RMSprop/embedding_88/embeddings/rmsRMSprop/dense_78/kernel/rmsRMSprop/dense_78/bias/rms*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_873789ś·
”
¦
I__inference_sequential_91_layer_call_and_return_conditional_losses_873412
embedding_88_input
embedding_88_873337
dense_78_873406
dense_78_873408
identity¢ dense_78/StatefulPartitionedCall¢"dropout_73/StatefulPartitionedCall¢$embedding_88/StatefulPartitionedCall”
$embedding_88/StatefulPartitionedCallStatefulPartitionedCallembedding_88_inputembedding_88_873337*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’2U*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_88_layer_call_and_return_conditional_losses_8733282&
$embedding_88/StatefulPartitionedCall
"dropout_73/StatefulPartitionedCallStatefulPartitionedCall-embedding_88/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’2U* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_73_layer_call_and_return_conditional_losses_8733522$
"dropout_73/StatefulPartitionedCall
flatten_80/PartitionedCallPartitionedCall+dropout_73/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’!* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_80_layer_call_and_return_conditional_losses_8733762
flatten_80/PartitionedCall±
 dense_78/StatefulPartitionedCallStatefulPartitionedCall#flatten_80/PartitionedCall:output:0dense_78_873406dense_78_873408*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_78_layer_call_and_return_conditional_losses_8733952"
 dense_78/StatefulPartitionedCallģ
IdentityIdentity)dense_78/StatefulPartitionedCall:output:0!^dense_78/StatefulPartitionedCall#^dropout_73/StatefulPartitionedCall%^embedding_88/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’2:::2D
 dense_78/StatefulPartitionedCall dense_78/StatefulPartitionedCall2H
"dropout_73/StatefulPartitionedCall"dropout_73/StatefulPartitionedCall2L
$embedding_88/StatefulPartitionedCall$embedding_88/StatefulPartitionedCall:[ W
'
_output_shapes
:’’’’’’’’’2
,
_user_specified_nameembedding_88_input
¦
G
+__inference_dropout_73_layer_call_fn_873611

inputs
identityČ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’2U* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_73_layer_call_and_return_conditional_losses_8733572
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:’’’’’’’’’2U2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’2U:S O
+
_output_shapes
:’’’’’’’’’2U
 
_user_specified_nameinputs


.__inference_sequential_91_layer_call_fn_873567

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_91_layer_call_and_return_conditional_losses_8734682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’2:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs


!__inference__wrapped_model_873314
embedding_88_input6
2sequential_91_embedding_88_embedding_lookup_8732989
5sequential_91_dense_78_matmul_readvariableop_resource:
6sequential_91_dense_78_biasadd_readvariableop_resource
identity¢-sequential_91/dense_78/BiasAdd/ReadVariableOp¢,sequential_91/dense_78/MatMul/ReadVariableOp¢+sequential_91/embedding_88/embedding_lookup
sequential_91/embedding_88/CastCastembedding_88_input*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’22!
sequential_91/embedding_88/Cast
+sequential_91/embedding_88/embedding_lookupResourceGather2sequential_91_embedding_88_embedding_lookup_873298#sequential_91/embedding_88/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*E
_class;
97loc:@sequential_91/embedding_88/embedding_lookup/873298*+
_output_shapes
:’’’’’’’’’2U*
dtype02-
+sequential_91/embedding_88/embedding_lookupŁ
4sequential_91/embedding_88/embedding_lookup/IdentityIdentity4sequential_91/embedding_88/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*E
_class;
97loc:@sequential_91/embedding_88/embedding_lookup/873298*+
_output_shapes
:’’’’’’’’’2U26
4sequential_91/embedding_88/embedding_lookup/Identityń
6sequential_91/embedding_88/embedding_lookup/Identity_1Identity=sequential_91/embedding_88/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:’’’’’’’’’2U28
6sequential_91/embedding_88/embedding_lookup/Identity_1É
!sequential_91/dropout_73/IdentityIdentity?sequential_91/embedding_88/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’2U2#
!sequential_91/dropout_73/Identity
sequential_91/flatten_80/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  2 
sequential_91/flatten_80/Const×
 sequential_91/flatten_80/ReshapeReshape*sequential_91/dropout_73/Identity:output:0'sequential_91/flatten_80/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’!2"
 sequential_91/flatten_80/ReshapeÓ
,sequential_91/dense_78/MatMul/ReadVariableOpReadVariableOp5sequential_91_dense_78_matmul_readvariableop_resource*
_output_shapes
:	!*
dtype02.
,sequential_91/dense_78/MatMul/ReadVariableOpŪ
sequential_91/dense_78/MatMulMatMul)sequential_91/flatten_80/Reshape:output:04sequential_91/dense_78/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_91/dense_78/MatMulŃ
-sequential_91/dense_78/BiasAdd/ReadVariableOpReadVariableOp6sequential_91_dense_78_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_91/dense_78/BiasAdd/ReadVariableOpŻ
sequential_91/dense_78/BiasAddBiasAdd'sequential_91/dense_78/MatMul:product:05sequential_91/dense_78/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2 
sequential_91/dense_78/BiasAdd¦
sequential_91/dense_78/SigmoidSigmoid'sequential_91/dense_78/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2 
sequential_91/dense_78/Sigmoid
IdentityIdentity"sequential_91/dense_78/Sigmoid:y:0.^sequential_91/dense_78/BiasAdd/ReadVariableOp-^sequential_91/dense_78/MatMul/ReadVariableOp,^sequential_91/embedding_88/embedding_lookup*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’2:::2^
-sequential_91/dense_78/BiasAdd/ReadVariableOp-sequential_91/dense_78/BiasAdd/ReadVariableOp2\
,sequential_91/dense_78/MatMul/ReadVariableOp,sequential_91/dense_78/MatMul/ReadVariableOp2Z
+sequential_91/embedding_88/embedding_lookup+sequential_91/embedding_88/embedding_lookup:[ W
'
_output_shapes
:’’’’’’’’’2
,
_user_specified_nameembedding_88_input
Ž
~
)__inference_dense_78_layer_call_fn_873642

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallō
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_78_layer_call_and_return_conditional_losses_8733952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’!::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’!
 
_user_specified_nameinputs
£
e
F__inference_dropout_73_layer_call_and_return_conditional_losses_873352

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:’’’’’’’’’2U2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeø
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’2U*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *?2
dropout/GreaterEqual/yĀ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:’’’’’’’’’2U2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:’’’’’’’’’2U2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:’’’’’’’’’2U2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:’’’’’’’’’2U2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’2U:S O
+
_output_shapes
:’’’’’’’’’2U
 
_user_specified_nameinputs
ą	

H__inference_embedding_88_layer_call_and_return_conditional_losses_873328

inputs
embedding_lookup_873322
identity¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’22
Castż
embedding_lookupResourceGatherembedding_lookup_873322Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0**
_class 
loc:@embedding_lookup/873322*+
_output_shapes
:’’’’’’’’’2U*
dtype02
embedding_lookupķ
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@embedding_lookup/873322*+
_output_shapes
:’’’’’’’’’2U2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:’’’’’’’’’2U2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:’’’’’’’’’2U2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’2:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
ēQ
č	
"__inference__traced_restore_873789
file_prefix,
(assignvariableop_embedding_88_embeddings&
"assignvariableop_1_dense_78_kernel$
 assignvariableop_2_dense_78_bias#
assignvariableop_3_rmsprop_iter$
 assignvariableop_4_rmsprop_decay,
(assignvariableop_5_rmsprop_learning_rate'
#assignvariableop_6_rmsprop_momentum"
assignvariableop_7_rmsprop_rho
assignvariableop_8_total
assignvariableop_9_count
assignvariableop_10_total_1
assignvariableop_11_count_1&
"assignvariableop_12_true_positives'
#assignvariableop_13_false_positives(
$assignvariableop_14_true_positives_1'
#assignvariableop_15_false_negatives;
7assignvariableop_16_rmsprop_embedding_88_embeddings_rms3
/assignvariableop_17_rmsprop_dense_78_kernel_rms1
-assignvariableop_18_rmsprop_dense_78_bias_rms
identity_20¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*	
value	B	B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¶
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity§
AssignVariableOpAssignVariableOp(assignvariableop_embedding_88_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1§
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_78_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2„
AssignVariableOp_2AssignVariableOp assignvariableop_2_dense_78_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_3¤
AssignVariableOp_3AssignVariableOpassignvariableop_3_rmsprop_iterIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4„
AssignVariableOp_4AssignVariableOp assignvariableop_4_rmsprop_decayIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5­
AssignVariableOp_5AssignVariableOp(assignvariableop_5_rmsprop_learning_rateIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Ø
AssignVariableOp_6AssignVariableOp#assignvariableop_6_rmsprop_momentumIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7£
AssignVariableOp_7AssignVariableOpassignvariableop_7_rmsprop_rhoIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOpassignvariableop_8_totalIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOpassignvariableop_9_countIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10£
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11£
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ŗ
AssignVariableOp_12AssignVariableOp"assignvariableop_12_true_positivesIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13«
AssignVariableOp_13AssignVariableOp#assignvariableop_13_false_positivesIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¬
AssignVariableOp_14AssignVariableOp$assignvariableop_14_true_positives_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15«
AssignVariableOp_15AssignVariableOp#assignvariableop_15_false_negativesIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16æ
AssignVariableOp_16AssignVariableOp7assignvariableop_16_rmsprop_embedding_88_embeddings_rmsIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17·
AssignVariableOp_17AssignVariableOp/assignvariableop_17_rmsprop_dense_78_kernel_rmsIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18µ
AssignVariableOp_18AssignVariableOp-assignvariableop_18_rmsprop_dense_78_bias_rmsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_189
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_19Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_19ó
Identity_20IdentityIdentity_19:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_20"#
identity_20Identity_20:output:0*a
_input_shapesP
N: :::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ł
d
F__inference_dropout_73_layer_call_and_return_conditional_losses_873601

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:’’’’’’’’’2U2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:’’’’’’’’’2U2

Identity_1"!

identity_1Identity_1:output:0**
_input_shapes
:’’’’’’’’’2U:S O
+
_output_shapes
:’’’’’’’’’2U
 
_user_specified_nameinputs
Ł
d
F__inference_dropout_73_layer_call_and_return_conditional_losses_873357

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:’’’’’’’’’2U2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:’’’’’’’’’2U2

Identity_1"!

identity_1Identity_1:output:0**
_input_shapes
:’’’’’’’’’2U:S O
+
_output_shapes
:’’’’’’’’’2U
 
_user_specified_nameinputs
É
õ
I__inference_sequential_91_layer_call_and_return_conditional_losses_873468

inputs
embedding_88_873457
dense_78_873462
dense_78_873464
identity¢ dense_78/StatefulPartitionedCall¢$embedding_88/StatefulPartitionedCall
$embedding_88/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_88_873457*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’2U*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_88_layer_call_and_return_conditional_losses_8733282&
$embedding_88/StatefulPartitionedCall
dropout_73/PartitionedCallPartitionedCall-embedding_88/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’2U* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_73_layer_call_and_return_conditional_losses_8733572
dropout_73/PartitionedCallų
flatten_80/PartitionedCallPartitionedCall#dropout_73/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’!* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_80_layer_call_and_return_conditional_losses_8733762
flatten_80/PartitionedCall±
 dense_78/StatefulPartitionedCallStatefulPartitionedCall#flatten_80/PartitionedCall:output:0dense_78_873462dense_78_873464*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_78_layer_call_and_return_conditional_losses_8733952"
 dense_78/StatefulPartitionedCallĒ
IdentityIdentity)dense_78/StatefulPartitionedCall:output:0!^dense_78/StatefulPartitionedCall%^embedding_88/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’2:::2D
 dense_78/StatefulPartitionedCall dense_78/StatefulPartitionedCall2L
$embedding_88/StatefulPartitionedCall$embedding_88/StatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
²
d
+__inference_dropout_73_layer_call_fn_873606

inputs
identity¢StatefulPartitionedCallą
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’2U* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_73_layer_call_and_return_conditional_losses_8733522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:’’’’’’’’’2U2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’2U22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:’’’’’’’’’2U
 
_user_specified_nameinputs
¶
b
F__inference_flatten_80_layer_call_and_return_conditional_losses_873376

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:’’’’’’’’’!2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:’’’’’’’’’!2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’2U:S O
+
_output_shapes
:’’’’’’’’’2U
 
_user_specified_nameinputs
Ģ
s
-__inference_embedding_88_layer_call_fn_873584

inputs
unknown
identity¢StatefulPartitionedCallļ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’2U*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_88_layer_call_and_return_conditional_losses_8733282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:’’’’’’’’’2U2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’2:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs


.__inference_sequential_91_layer_call_fn_873556

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_91_layer_call_and_return_conditional_losses_8734432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’2:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
ŗ/

__inference__traced_save_873722
file_prefix6
2savev2_embedding_88_embeddings_read_readvariableop.
*savev2_dense_78_kernel_read_readvariableop,
(savev2_dense_78_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop-
)savev2_true_positives_read_readvariableop.
*savev2_false_positives_read_readvariableop/
+savev2_true_positives_1_read_readvariableop.
*savev2_false_negatives_read_readvariableopB
>savev2_rmsprop_embedding_88_embeddings_rms_read_readvariableop:
6savev2_rmsprop_dense_78_kernel_rms_read_readvariableop8
4savev2_rmsprop_dense_78_bias_rms_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*	
value	B	B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names°
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_embedding_88_embeddings_read_readvariableop*savev2_dense_78_kernel_read_readvariableop(savev2_dense_78_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableop+savev2_true_positives_1_read_readvariableop*savev2_false_negatives_read_readvariableop>savev2_rmsprop_embedding_88_embeddings_rms_read_readvariableop6savev2_rmsprop_dense_78_kernel_rms_read_readvariableop4savev2_rmsprop_dense_78_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *"
dtypes
2	2
SaveV2ŗ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes”
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*y
_input_shapesh
f: :	½{U:	!:: : : : : : : : : :::::	½{U:	!:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	½{U:%!

_output_shapes
:	!: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	½{U:%!

_output_shapes
:	!: 

_output_shapes
::

_output_shapes
: 
ż

I__inference_sequential_91_layer_call_and_return_conditional_losses_873443

inputs
embedding_88_873432
dense_78_873437
dense_78_873439
identity¢ dense_78/StatefulPartitionedCall¢"dropout_73/StatefulPartitionedCall¢$embedding_88/StatefulPartitionedCall
$embedding_88/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_88_873432*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’2U*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_88_layer_call_and_return_conditional_losses_8733282&
$embedding_88/StatefulPartitionedCall
"dropout_73/StatefulPartitionedCallStatefulPartitionedCall-embedding_88/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’2U* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_73_layer_call_and_return_conditional_losses_8733522$
"dropout_73/StatefulPartitionedCall
flatten_80/PartitionedCallPartitionedCall+dropout_73/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’!* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_80_layer_call_and_return_conditional_losses_8733762
flatten_80/PartitionedCall±
 dense_78/StatefulPartitionedCallStatefulPartitionedCall#flatten_80/PartitionedCall:output:0dense_78_873437dense_78_873439*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_78_layer_call_and_return_conditional_losses_8733952"
 dense_78/StatefulPartitionedCallģ
IdentityIdentity)dense_78/StatefulPartitionedCall:output:0!^dense_78/StatefulPartitionedCall#^dropout_73/StatefulPartitionedCall%^embedding_88/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’2:::2D
 dense_78/StatefulPartitionedCall dense_78/StatefulPartitionedCall2H
"dropout_73/StatefulPartitionedCall"dropout_73/StatefulPartitionedCall2L
$embedding_88/StatefulPartitionedCall$embedding_88/StatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
ą	

H__inference_embedding_88_layer_call_and_return_conditional_losses_873577

inputs
embedding_lookup_873571
identity¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’22
Castż
embedding_lookupResourceGatherembedding_lookup_873571Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0**
_class 
loc:@embedding_lookup/873571*+
_output_shapes
:’’’’’’’’’2U*
dtype02
embedding_lookupķ
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@embedding_lookup/873571*+
_output_shapes
:’’’’’’’’’2U2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:’’’’’’’’’2U2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:’’’’’’’’’2U2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’2:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
«

.__inference_sequential_91_layer_call_fn_873477
embedding_88_input
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallembedding_88_inputunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_91_layer_call_and_return_conditional_losses_8734682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’2:::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
'
_output_shapes
:’’’’’’’’’2
,
_user_specified_nameembedding_88_input
ó	
Ż
D__inference_dense_78_layer_call_and_return_conditional_losses_873633

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	!*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’!::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’!
 
_user_specified_nameinputs
«

.__inference_sequential_91_layer_call_fn_873452
embedding_88_input
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallembedding_88_inputunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_91_layer_call_and_return_conditional_losses_8734432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’2:::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
'
_output_shapes
:’’’’’’’’’2
,
_user_specified_nameembedding_88_input
ó	
Ż
D__inference_dense_78_layer_call_and_return_conditional_losses_873395

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	!*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’!::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’!
 
_user_specified_nameinputs
 
G
+__inference_flatten_80_layer_call_fn_873622

inputs
identityÅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’!* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_80_layer_call_and_return_conditional_losses_8733762
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:’’’’’’’’’!2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’2U:S O
+
_output_shapes
:’’’’’’’’’2U
 
_user_specified_nameinputs

Š
I__inference_sequential_91_layer_call_and_return_conditional_losses_873545

inputs(
$embedding_88_embedding_lookup_873529+
'dense_78_matmul_readvariableop_resource,
(dense_78_biasadd_readvariableop_resource
identity¢dense_78/BiasAdd/ReadVariableOp¢dense_78/MatMul/ReadVariableOp¢embedding_88/embedding_lookupw
embedding_88/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’22
embedding_88/Cast¾
embedding_88/embedding_lookupResourceGather$embedding_88_embedding_lookup_873529embedding_88/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*7
_class-
+)loc:@embedding_88/embedding_lookup/873529*+
_output_shapes
:’’’’’’’’’2U*
dtype02
embedding_88/embedding_lookup”
&embedding_88/embedding_lookup/IdentityIdentity&embedding_88/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_88/embedding_lookup/873529*+
_output_shapes
:’’’’’’’’’2U2(
&embedding_88/embedding_lookup/IdentityĒ
(embedding_88/embedding_lookup/Identity_1Identity/embedding_88/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:’’’’’’’’’2U2*
(embedding_88/embedding_lookup/Identity_1
dropout_73/IdentityIdentity1embedding_88/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’2U2
dropout_73/Identityu
flatten_80/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  2
flatten_80/Const
flatten_80/ReshapeReshapedropout_73/Identity:output:0flatten_80/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’!2
flatten_80/Reshape©
dense_78/MatMul/ReadVariableOpReadVariableOp'dense_78_matmul_readvariableop_resource*
_output_shapes
:	!*
dtype02 
dense_78/MatMul/ReadVariableOp£
dense_78/MatMulMatMulflatten_80/Reshape:output:0&dense_78/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_78/MatMul§
dense_78/BiasAdd/ReadVariableOpReadVariableOp(dense_78_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_78/BiasAdd/ReadVariableOp„
dense_78/BiasAddBiasAdddense_78/MatMul:product:0'dense_78/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_78/BiasAdd|
dense_78/SigmoidSigmoiddense_78/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_78/SigmoidĖ
IdentityIdentitydense_78/Sigmoid:y:0 ^dense_78/BiasAdd/ReadVariableOp^dense_78/MatMul/ReadVariableOp^embedding_88/embedding_lookup*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’2:::2B
dense_78/BiasAdd/ReadVariableOpdense_78/BiasAdd/ReadVariableOp2@
dense_78/MatMul/ReadVariableOpdense_78/MatMul/ReadVariableOp2>
embedding_88/embedding_lookupembedding_88/embedding_lookup:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
ķ

I__inference_sequential_91_layer_call_and_return_conditional_losses_873426
embedding_88_input
embedding_88_873415
dense_78_873420
dense_78_873422
identity¢ dense_78/StatefulPartitionedCall¢$embedding_88/StatefulPartitionedCall”
$embedding_88/StatefulPartitionedCallStatefulPartitionedCallembedding_88_inputembedding_88_873415*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’2U*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_88_layer_call_and_return_conditional_losses_8733282&
$embedding_88/StatefulPartitionedCall
dropout_73/PartitionedCallPartitionedCall-embedding_88/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’2U* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_73_layer_call_and_return_conditional_losses_8733572
dropout_73/PartitionedCallų
flatten_80/PartitionedCallPartitionedCall#dropout_73/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’!* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_80_layer_call_and_return_conditional_losses_8733762
flatten_80/PartitionedCall±
 dense_78/StatefulPartitionedCallStatefulPartitionedCall#flatten_80/PartitionedCall:output:0dense_78_873420dense_78_873422*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_78_layer_call_and_return_conditional_losses_8733952"
 dense_78/StatefulPartitionedCallĒ
IdentityIdentity)dense_78/StatefulPartitionedCall:output:0!^dense_78/StatefulPartitionedCall%^embedding_88/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’2:::2D
 dense_78/StatefulPartitionedCall dense_78/StatefulPartitionedCall2L
$embedding_88/StatefulPartitionedCall$embedding_88/StatefulPartitionedCall:[ W
'
_output_shapes
:’’’’’’’’’2
,
_user_specified_nameembedding_88_input
ł

$__inference_signature_wrapper_873498
embedding_88_input
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCallź
StatefulPartitionedCallStatefulPartitionedCallembedding_88_inputunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_8733142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’2:::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
'
_output_shapes
:’’’’’’’’’2
,
_user_specified_nameembedding_88_input
£
e
F__inference_dropout_73_layer_call_and_return_conditional_losses_873596

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:’’’’’’’’’2U2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeø
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’2U*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *?2
dropout/GreaterEqual/yĀ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:’’’’’’’’’2U2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:’’’’’’’’’2U2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:’’’’’’’’’2U2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:’’’’’’’’’2U2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’2U:S O
+
_output_shapes
:’’’’’’’’’2U
 
_user_specified_nameinputs
¶
b
F__inference_flatten_80_layer_call_and_return_conditional_losses_873617

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:’’’’’’’’’!2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:’’’’’’’’’!2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’2U:S O
+
_output_shapes
:’’’’’’’’’2U
 
_user_specified_nameinputs
ö!
Š
I__inference_sequential_91_layer_call_and_return_conditional_losses_873525

inputs(
$embedding_88_embedding_lookup_873502+
'dense_78_matmul_readvariableop_resource,
(dense_78_biasadd_readvariableop_resource
identity¢dense_78/BiasAdd/ReadVariableOp¢dense_78/MatMul/ReadVariableOp¢embedding_88/embedding_lookupw
embedding_88/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’22
embedding_88/Cast¾
embedding_88/embedding_lookupResourceGather$embedding_88_embedding_lookup_873502embedding_88/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*7
_class-
+)loc:@embedding_88/embedding_lookup/873502*+
_output_shapes
:’’’’’’’’’2U*
dtype02
embedding_88/embedding_lookup”
&embedding_88/embedding_lookup/IdentityIdentity&embedding_88/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_88/embedding_lookup/873502*+
_output_shapes
:’’’’’’’’’2U2(
&embedding_88/embedding_lookup/IdentityĒ
(embedding_88/embedding_lookup/Identity_1Identity/embedding_88/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:’’’’’’’’’2U2*
(embedding_88/embedding_lookup/Identity_1y
dropout_73/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_73/dropout/ConstĆ
dropout_73/dropout/MulMul1embedding_88/embedding_lookup/Identity_1:output:0!dropout_73/dropout/Const:output:0*
T0*+
_output_shapes
:’’’’’’’’’2U2
dropout_73/dropout/Mul
dropout_73/dropout/ShapeShape1embedding_88/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
dropout_73/dropout/ShapeŁ
/dropout_73/dropout/random_uniform/RandomUniformRandomUniform!dropout_73/dropout/Shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’2U*
dtype021
/dropout_73/dropout/random_uniform/RandomUniform
!dropout_73/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *?2#
!dropout_73/dropout/GreaterEqual/yī
dropout_73/dropout/GreaterEqualGreaterEqual8dropout_73/dropout/random_uniform/RandomUniform:output:0*dropout_73/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:’’’’’’’’’2U2!
dropout_73/dropout/GreaterEqual¤
dropout_73/dropout/CastCast#dropout_73/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:’’’’’’’’’2U2
dropout_73/dropout/CastŖ
dropout_73/dropout/Mul_1Muldropout_73/dropout/Mul:z:0dropout_73/dropout/Cast:y:0*
T0*+
_output_shapes
:’’’’’’’’’2U2
dropout_73/dropout/Mul_1u
flatten_80/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  2
flatten_80/Const
flatten_80/ReshapeReshapedropout_73/dropout/Mul_1:z:0flatten_80/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’!2
flatten_80/Reshape©
dense_78/MatMul/ReadVariableOpReadVariableOp'dense_78_matmul_readvariableop_resource*
_output_shapes
:	!*
dtype02 
dense_78/MatMul/ReadVariableOp£
dense_78/MatMulMatMulflatten_80/Reshape:output:0&dense_78/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_78/MatMul§
dense_78/BiasAdd/ReadVariableOpReadVariableOp(dense_78_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_78/BiasAdd/ReadVariableOp„
dense_78/BiasAddBiasAdddense_78/MatMul:product:0'dense_78/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_78/BiasAdd|
dense_78/SigmoidSigmoiddense_78/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_78/SigmoidĖ
IdentityIdentitydense_78/Sigmoid:y:0 ^dense_78/BiasAdd/ReadVariableOp^dense_78/MatMul/ReadVariableOp^embedding_88/embedding_lookup*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’2:::2B
dense_78/BiasAdd/ReadVariableOpdense_78/BiasAdd/ReadVariableOp2@
dense_78/MatMul/ReadVariableOpdense_78/MatMul/ReadVariableOp2>
embedding_88/embedding_lookupembedding_88/embedding_lookup:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Į
serving_default­
Q
embedding_88_input;
$serving_default_embedding_88_input:0’’’’’’’’’2<
dense_780
StatefulPartitionedCall:0’’’’’’’’’tensorflow/serving/predict:ķ
¢!
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
	optimizer
trainable_variables
regularization_losses
	variables
		keras_api


signatures
*V&call_and_return_all_conditional_losses
W_default_save_signature
X__call__"ļ
_tf_keras_sequentialŠ{"class_name": "Sequential", "name": "sequential_91", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_91", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "embedding_88_input"}}, {"class_name": "Embedding", "config": {"name": "embedding_88", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50]}, "dtype": "float32", "input_dim": 15805, "output_dim": 85, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 50}}, {"class_name": "Dropout", "config": {"name": "dropout_73", "trainable": true, "dtype": "float32", "rate": 0.6, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_80", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_78", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_91", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "embedding_88_input"}}, {"class_name": "Embedding", "config": {"name": "embedding_88", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50]}, "dtype": "float32", "input_dim": 15805, "output_dim": 85, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 50}}, {"class_name": "Dropout", "config": {"name": "dropout_73", "trainable": true, "dtype": "float32", "rate": 0.6, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_80", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_78", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}, {"class_name": "Precision", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}, {"class_name": "Recall", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
«

embeddings
trainable_variables
regularization_losses
	variables
	keras_api
*Y&call_and_return_all_conditional_losses
Z__call__"
_tf_keras_layerņ{"class_name": "Embedding", "name": "embedding_88", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_88", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50]}, "dtype": "float32", "input_dim": 15805, "output_dim": 85, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 50}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
ē
trainable_variables
regularization_losses
	variables
	keras_api
*[&call_and_return_all_conditional_losses
\__call__"Ų
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_73", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_73", "trainable": true, "dtype": "float32", "rate": 0.6, "noise_shape": null, "seed": null}}
č
trainable_variables
regularization_losses
	variables
	keras_api
*]&call_and_return_all_conditional_losses
^__call__"Ł
_tf_keras_layeræ{"class_name": "Flatten", "name": "flatten_80", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_80", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ų

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*_&call_and_return_all_conditional_losses
`__call__"Ó
_tf_keras_layer¹{"class_name": "Dense", "name": "dense_78", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_78", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4250}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4250]}}
s
iter
	decay
 learning_rate
!momentum
"rho	rmsS	rmsT	rmsU"
	optimizer
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
Ź
#metrics
trainable_variables
$layer_regularization_losses
regularization_losses
	variables
%non_trainable_variables
&layer_metrics

'layers
X__call__
W_default_save_signature
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
,
aserving_default"
signature_map
*:(	½{U2embedding_88/embeddings
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
­
(metrics
trainable_variables
)layer_regularization_losses
regularization_losses
	variables
*non_trainable_variables
+layer_metrics

,layers
Z__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
-metrics
trainable_variables
.layer_regularization_losses
regularization_losses
	variables
/non_trainable_variables
0layer_metrics

1layers
\__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
2metrics
trainable_variables
3layer_regularization_losses
regularization_losses
	variables
4non_trainable_variables
5layer_metrics

6layers
^__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
": 	!2dense_78/kernel
:2dense_78/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
7metrics
trainable_variables
8layer_regularization_losses
regularization_losses
	variables
9non_trainable_variables
:layer_metrics

;layers
`__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
<
<0
=1
>2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
»
	@total
	Acount
B	variables
C	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
ś
	Dtotal
	Ecount
F
_fn_kwargs
G	variables
H	keras_api"³
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
£
I
thresholds
Jtrue_positives
Kfalse_positives
L	variables
M	keras_api"É
_tf_keras_metric®{"class_name": "Precision", "name": "precision", "dtype": "float32", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}

N
thresholds
Otrue_positives
Pfalse_negatives
Q	variables
R	keras_api"Ą
_tf_keras_metric„{"class_name": "Recall", "name": "recall", "dtype": "float32", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
:  (2total
:  (2count
.
@0
A1"
trackable_list_wrapper
-
B	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
D0
E1"
trackable_list_wrapper
-
G	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
.
J0
K1"
trackable_list_wrapper
-
L	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
.
O0
P1"
trackable_list_wrapper
-
Q	variables"
_generic_user_object
4:2	½{U2#RMSprop/embedding_88/embeddings/rms
,:*	!2RMSprop/dense_78/kernel/rms
%:#2RMSprop/dense_78/bias/rms
ņ2ļ
I__inference_sequential_91_layer_call_and_return_conditional_losses_873412
I__inference_sequential_91_layer_call_and_return_conditional_losses_873545
I__inference_sequential_91_layer_call_and_return_conditional_losses_873426
I__inference_sequential_91_layer_call_and_return_conditional_losses_873525Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ź2ē
!__inference__wrapped_model_873314Į
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *1¢.
,)
embedding_88_input’’’’’’’’’2
2
.__inference_sequential_91_layer_call_fn_873452
.__inference_sequential_91_layer_call_fn_873556
.__inference_sequential_91_layer_call_fn_873477
.__inference_sequential_91_layer_call_fn_873567Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ņ2ļ
H__inference_embedding_88_layer_call_and_return_conditional_losses_873577¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
×2Ō
-__inference_embedding_88_layer_call_fn_873584¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ź2Ē
F__inference_dropout_73_layer_call_and_return_conditional_losses_873596
F__inference_dropout_73_layer_call_and_return_conditional_losses_873601“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
+__inference_dropout_73_layer_call_fn_873606
+__inference_dropout_73_layer_call_fn_873611“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
š2ķ
F__inference_flatten_80_layer_call_and_return_conditional_losses_873617¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Õ2Ņ
+__inference_flatten_80_layer_call_fn_873622¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ī2ė
D__inference_dense_78_layer_call_and_return_conditional_losses_873633¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ó2Š
)__inference_dense_78_layer_call_fn_873642¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ÖBÓ
$__inference_signature_wrapper_873498embedding_88_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
!__inference__wrapped_model_873314w;¢8
1¢.
,)
embedding_88_input’’’’’’’’’2
Ŗ "3Ŗ0
.
dense_78"
dense_78’’’’’’’’’„
D__inference_dense_78_layer_call_and_return_conditional_losses_873633]0¢-
&¢#
!
inputs’’’’’’’’’!
Ŗ "%¢"

0’’’’’’’’’
 }
)__inference_dense_78_layer_call_fn_873642P0¢-
&¢#
!
inputs’’’’’’’’’!
Ŗ "’’’’’’’’’®
F__inference_dropout_73_layer_call_and_return_conditional_losses_873596d7¢4
-¢*
$!
inputs’’’’’’’’’2U
p
Ŗ ")¢&

0’’’’’’’’’2U
 ®
F__inference_dropout_73_layer_call_and_return_conditional_losses_873601d7¢4
-¢*
$!
inputs’’’’’’’’’2U
p 
Ŗ ")¢&

0’’’’’’’’’2U
 
+__inference_dropout_73_layer_call_fn_873606W7¢4
-¢*
$!
inputs’’’’’’’’’2U
p
Ŗ "’’’’’’’’’2U
+__inference_dropout_73_layer_call_fn_873611W7¢4
-¢*
$!
inputs’’’’’’’’’2U
p 
Ŗ "’’’’’’’’’2U«
H__inference_embedding_88_layer_call_and_return_conditional_losses_873577_/¢,
%¢"
 
inputs’’’’’’’’’2
Ŗ ")¢&

0’’’’’’’’’2U
 
-__inference_embedding_88_layer_call_fn_873584R/¢,
%¢"
 
inputs’’’’’’’’’2
Ŗ "’’’’’’’’’2U§
F__inference_flatten_80_layer_call_and_return_conditional_losses_873617]3¢0
)¢&
$!
inputs’’’’’’’’’2U
Ŗ "&¢#

0’’’’’’’’’!
 
+__inference_flatten_80_layer_call_fn_873622P3¢0
)¢&
$!
inputs’’’’’’’’’2U
Ŗ "’’’’’’’’’!¾
I__inference_sequential_91_layer_call_and_return_conditional_losses_873412qC¢@
9¢6
,)
embedding_88_input’’’’’’’’’2
p

 
Ŗ "%¢"

0’’’’’’’’’
 ¾
I__inference_sequential_91_layer_call_and_return_conditional_losses_873426qC¢@
9¢6
,)
embedding_88_input’’’’’’’’’2
p 

 
Ŗ "%¢"

0’’’’’’’’’
 ²
I__inference_sequential_91_layer_call_and_return_conditional_losses_873525e7¢4
-¢*
 
inputs’’’’’’’’’2
p

 
Ŗ "%¢"

0’’’’’’’’’
 ²
I__inference_sequential_91_layer_call_and_return_conditional_losses_873545e7¢4
-¢*
 
inputs’’’’’’’’’2
p 

 
Ŗ "%¢"

0’’’’’’’’’
 
.__inference_sequential_91_layer_call_fn_873452dC¢@
9¢6
,)
embedding_88_input’’’’’’’’’2
p

 
Ŗ "’’’’’’’’’
.__inference_sequential_91_layer_call_fn_873477dC¢@
9¢6
,)
embedding_88_input’’’’’’’’’2
p 

 
Ŗ "’’’’’’’’’
.__inference_sequential_91_layer_call_fn_873556X7¢4
-¢*
 
inputs’’’’’’’’’2
p

 
Ŗ "’’’’’’’’’
.__inference_sequential_91_layer_call_fn_873567X7¢4
-¢*
 
inputs’’’’’’’’’2
p 

 
Ŗ "’’’’’’’’’¶
$__inference_signature_wrapper_873498Q¢N
¢ 
GŖD
B
embedding_88_input,)
embedding_88_input’’’’’’’’’2"3Ŗ0
.
dense_78"
dense_78’’’’’’’’’