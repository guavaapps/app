Ë
Ô·
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
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

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
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
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
÷
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.10.02unknown8èÐ
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

lstmp_3/lstmp_cell_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namelstmp_3/lstmp_cell_3/bias

-lstmp_3/lstmp_cell_3/bias/Read/ReadVariableOpReadVariableOplstmp_3/lstmp_cell_3/bias*
_output_shapes	
:*
dtype0
§
%lstmp_3/lstmp_cell_3/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*6
shared_name'%lstmp_3/lstmp_cell_3/recurrent_kernel
 
9lstmp_3/lstmp_cell_3/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstmp_3/lstmp_cell_3/recurrent_kernel*
_output_shapes
:	@*
dtype0

lstmp_3/lstmp_cell_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*,
shared_namelstmp_3/lstmp_cell_3/kernel

/lstmp_3/lstmp_cell_3/kernel/Read/ReadVariableOpReadVariableOplstmp_3/lstmp_cell_3/kernel*
_output_shapes
:	@*
dtype0
h
proj_wVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_nameproj_w
a
proj_w/Read/ReadVariableOpReadVariableOpproj_w*
_output_shapes

:@@*
dtype0

lstmp_2/lstmp_cell_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namelstmp_2/lstmp_cell_2/bias

-lstmp_2/lstmp_cell_2/bias/Read/ReadVariableOpReadVariableOplstmp_2/lstmp_cell_2/bias*
_output_shapes	
:*
dtype0
§
%lstmp_2/lstmp_cell_2/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*6
shared_name'%lstmp_2/lstmp_cell_2/recurrent_kernel
 
9lstmp_2/lstmp_cell_2/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstmp_2/lstmp_cell_2/recurrent_kernel*
_output_shapes
:	@*
dtype0

lstmp_2/lstmp_cell_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*,
shared_namelstmp_2/lstmp_cell_2/kernel

/lstmp_2/lstmp_cell_2/kernel/Read/ReadVariableOpReadVariableOplstmp_2/lstmp_cell_2/kernel*
_output_shapes
:	*
dtype0
l
proj_w_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_name
proj_w_1
e
proj_w_1/Read/ReadVariableOpReadVariableOpproj_w_1*
_output_shapes

:@@*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@*
dtype0

serving_default_input_2Placeholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ
µ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2lstmp_2/lstmp_cell_2/kernel%lstmp_2/lstmp_cell_2/recurrent_kernellstmp_2/lstmp_cell_2/biasproj_w_1lstmp_3/lstmp_cell_3/kernel%lstmp_3/lstmp_cell_3/recurrent_kernellstmp_3/lstmp_cell_3/biasproj_wdense_1/kerneldense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_signature_wrapper_2127

NoOpNoOp
§*
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*â)
valueØ)BÕ) BÎ)
Î
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
ª
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
cell

state_spec*
ª
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
cell

state_spec*
¦
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias*
J
&0
'1
(2
)3
*4
+5
,6
-7
$8
%9*
J
&0
'1
(2
)3
*4
+5
,6
-7
$8
%9*
* 
°
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
6
3trace_0
4trace_1
5trace_2
6trace_3* 
6
7trace_0
8trace_1
9trace_2
:trace_3* 
* 
* 

;serving_default* 
 
&0
'1
(2
)3*
 
&0
'1
(2
)3*
* 


<states
=non_trainable_variables

>layers
?metrics
@layer_regularization_losses
Alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Btrace_0
Ctrace_1* 

Dtrace_0
Etrace_1* 
ï
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses
L_random_generator
M
state_size

&proj_w

'kernel
(recurrent_kernel
)bias*
* 
 
*0
+1
,2
-3*
 
*0
+1
,2
-3*
* 


Nstates
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ttrace_0
Utrace_1* 

Vtrace_0
Wtrace_1* 
ï
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses
^_random_generator
_
state_size

*proj_w

+kernel
,recurrent_kernel
-bias*
* 

$0
%1*

$0
%1*
* 

`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

etrace_0* 

ftrace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
HB
VARIABLE_VALUEproj_w_1&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstmp_2/lstmp_cell_2/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%lstmp_2/lstmp_cell_2/recurrent_kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstmp_2/lstmp_cell_2/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
F@
VARIABLE_VALUEproj_w&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstmp_3/lstmp_cell_3/kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%lstmp_3/lstmp_cell_3/recurrent_kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstmp_3/lstmp_cell_3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

g0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
 
&0
'1
(2
)3*
 
&0
'1
(2
)3*
* 

hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
 
*0
+1
,2
-3*
 
*0
+1
,2
-3*
* 

mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
r	variables
s	keras_api
	ttotal
	ucount*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

t0
u1*

r	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpproj_w_1/Read/ReadVariableOp/lstmp_2/lstmp_cell_2/kernel/Read/ReadVariableOp9lstmp_2/lstmp_cell_2/recurrent_kernel/Read/ReadVariableOp-lstmp_2/lstmp_cell_2/bias/Read/ReadVariableOpproj_w/Read/ReadVariableOp/lstmp_3/lstmp_cell_3/kernel/Read/ReadVariableOp9lstmp_3/lstmp_cell_3/recurrent_kernel/Read/ReadVariableOp-lstmp_3/lstmp_cell_3/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *&
f!R
__inference__traced_save_3527

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_1/kerneldense_1/biasproj_w_1lstmp_2/lstmp_cell_2/kernel%lstmp_2/lstmp_cell_2/recurrent_kernellstmp_2/lstmp_cell_2/biasproj_wlstmp_3/lstmp_cell_3/kernel%lstmp_3/lstmp_cell_3/recurrent_kernellstmp_3/lstmp_cell_3/biastotalcount*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_restore_3573òý
 
±
dlstmp_lstmp_2_while_cond_958:
6dlstmp_lstmp_2_while_dlstmp_lstmp_2_while_loop_counter@
<dlstmp_lstmp_2_while_dlstmp_lstmp_2_while_maximum_iterations$
 dlstmp_lstmp_2_while_placeholder&
"dlstmp_lstmp_2_while_placeholder_1&
"dlstmp_lstmp_2_while_placeholder_2&
"dlstmp_lstmp_2_while_placeholder_3<
8dlstmp_lstmp_2_while_less_dlstmp_lstmp_2_strided_slice_1O
Kdlstmp_lstmp_2_while_dlstmp_lstmp_2_while_cond_958___redundant_placeholder0O
Kdlstmp_lstmp_2_while_dlstmp_lstmp_2_while_cond_958___redundant_placeholder1O
Kdlstmp_lstmp_2_while_dlstmp_lstmp_2_while_cond_958___redundant_placeholder2O
Kdlstmp_lstmp_2_while_dlstmp_lstmp_2_while_cond_958___redundant_placeholder3O
Kdlstmp_lstmp_2_while_dlstmp_lstmp_2_while_cond_958___redundant_placeholder4!
dlstmp_lstmp_2_while_identity

dlstmp/lstmp_2/while/LessLess dlstmp_lstmp_2_while_placeholder8dlstmp_lstmp_2_while_less_dlstmp_lstmp_2_strided_slice_1*
T0*
_output_shapes
: i
dlstmp/lstmp_2/while/IdentityIdentitydlstmp/lstmp_2/while/Less:z:0*
T0
*
_output_shapes
: "G
dlstmp_lstmp_2_while_identity&dlstmp/lstmp_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
 
ÿ
@__inference_dlstmp_layer_call_and_return_conditional_losses_1547

inputs
lstmp_2_1359:	
lstmp_2_1361:	@
lstmp_2_1363:	
lstmp_2_1365:@@
lstmp_3_1521:	@
lstmp_3_1523:	@
lstmp_3_1525:	
lstmp_3_1527:@@
dense_1_1541:@
dense_1_1543:
identity¢dense_1/StatefulPartitionedCall¢lstmp_2/StatefulPartitionedCall¢lstmp_3/StatefulPartitionedCall
lstmp_2/StatefulPartitionedCallStatefulPartitionedCallinputslstmp_2_1359lstmp_2_1361lstmp_2_1363lstmp_2_1365*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstmp_2_layer_call_and_return_conditional_losses_1358«
lstmp_3/StatefulPartitionedCallStatefulPartitionedCall(lstmp_2/StatefulPartitionedCall:output:0lstmp_3_1521lstmp_3_1523lstmp_3_1525lstmp_3_1527*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstmp_3_layer_call_and_return_conditional_losses_1520
dense_1/StatefulPartitionedCallStatefulPartitionedCall(lstmp_3/StatefulPartitionedCall:output:0dense_1_1541dense_1_1543*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_1540w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
NoOpNoOp ^dense_1/StatefulPartitionedCall ^lstmp_2/StatefulPartitionedCall ^lstmp_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
lstmp_2/StatefulPartitionedCalllstmp_2/StatefulPartitionedCall2B
lstmp_3/StatefulPartitionedCalllstmp_3/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
í
while_cond_3205
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_12
.while_while_cond_3205___redundant_placeholder02
.while_while_cond_3205___redundant_placeholder12
.while_while_cond_3205___redundant_placeholder22
.while_while_cond_3205___redundant_placeholder32
.while_while_cond_3205___redundant_placeholder4
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
£

@__inference_dlstmp_layer_call_and_return_conditional_losses_2092
input_2
lstmp_2_2068:	
lstmp_2_2070:	@
lstmp_2_2072:	
lstmp_2_2074:@@
lstmp_3_2077:	@
lstmp_3_2079:	@
lstmp_3_2081:	
lstmp_3_2083:@@
dense_1_2086:@
dense_1_2088:
identity¢dense_1/StatefulPartitionedCall¢lstmp_2/StatefulPartitionedCall¢lstmp_3/StatefulPartitionedCall
lstmp_2/StatefulPartitionedCallStatefulPartitionedCallinput_2lstmp_2_2068lstmp_2_2070lstmp_2_2072lstmp_2_2074*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstmp_2_layer_call_and_return_conditional_losses_1925«
lstmp_3/StatefulPartitionedCallStatefulPartitionedCall(lstmp_2/StatefulPartitionedCall:output:0lstmp_3_2077lstmp_3_2079lstmp_3_2081lstmp_3_2083*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstmp_3_layer_call_and_return_conditional_losses_1748
dense_1/StatefulPartitionedCallStatefulPartitionedCall(lstmp_3/StatefulPartitionedCall:output:0dense_1_2086dense_1_2088*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_1540w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
NoOpNoOp ^dense_1/StatefulPartitionedCall ^lstmp_2/StatefulPartitionedCall ^lstmp_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
lstmp_2/StatefulPartitionedCalllstmp_2/StatefulPartitionedCall2B
lstmp_3/StatefulPartitionedCalllstmp_3/StatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
ìN

A__inference_lstmp_3_layer_call_and_return_conditional_losses_1748

inputs>
+lstmp_cell_3_matmul_readvariableop_resource:	@@
-lstmp_cell_3_matmul_1_readvariableop_resource:	@;
,lstmp_cell_3_biasadd_readvariableop_resource:	?
-lstmp_cell_3_matmul_2_readvariableop_resource:@@
identity¢#lstmp_cell_3/BiasAdd/ReadVariableOp¢"lstmp_cell_3/MatMul/ReadVariableOp¢$lstmp_cell_3/MatMul_1/ReadVariableOp¢$lstmp_cell_3/MatMul_2/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
"lstmp_cell_3/MatMul/ReadVariableOpReadVariableOp+lstmp_cell_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
lstmp_cell_3/MatMulMatMulstrided_slice_2:output:0*lstmp_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstmp_cell_3/MatMul_1/ReadVariableOpReadVariableOp-lstmp_cell_3_matmul_1_readvariableop_resource*
_output_shapes
:	@*
dtype0
lstmp_cell_3/MatMul_1MatMulzeros:output:0,lstmp_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstmp_cell_3/addAddV2lstmp_cell_3/MatMul:product:0lstmp_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstmp_cell_3/BiasAdd/ReadVariableOpReadVariableOp,lstmp_cell_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstmp_cell_3/BiasAddBiasAddlstmp_cell_3/add:z:0+lstmp_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstmp_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstmp_cell_3/splitSplit%lstmp_cell_3/split/split_dim:output:0lstmp_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*
	num_splitn
lstmp_cell_3/SigmoidSigmoidlstmp_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@p
lstmp_cell_3/Sigmoid_1Sigmoidlstmp_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
lstmp_cell_3/mulMullstmp_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
lstmp_cell_3/TanhTanhlstmp_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
lstmp_cell_3/mul_1Mullstmp_cell_3/Sigmoid:y:0lstmp_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
lstmp_cell_3/add_1AddV2lstmp_cell_3/mul:z:0lstmp_cell_3/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@p
lstmp_cell_3/Sigmoid_2Sigmoidlstmp_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
lstmp_cell_3/Tanh_1Tanhlstmp_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_cell_3/mul_2Mullstmp_cell_3/Sigmoid_2:y:0lstmp_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
$lstmp_cell_3/MatMul_2/ReadVariableOpReadVariableOp-lstmp_cell_3_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype0
lstmp_cell_3/MatMul_2MatMulzeros:output:0,lstmp_cell_3/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ³
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstmp_cell_3_matmul_readvariableop_resource-lstmp_cell_3_matmul_1_readvariableop_resource,lstmp_cell_3_biasadd_readvariableop_resource-lstmp_cell_3_matmul_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *&
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1659*
condR
while_cond_1658*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ç
NoOpNoOp$^lstmp_cell_3/BiasAdd/ReadVariableOp#^lstmp_cell_3/MatMul/ReadVariableOp%^lstmp_cell_3/MatMul_1/ReadVariableOp%^lstmp_cell_3/MatMul_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : : : 2J
#lstmp_cell_3/BiasAdd/ReadVariableOp#lstmp_cell_3/BiasAdd/ReadVariableOp2H
"lstmp_cell_3/MatMul/ReadVariableOp"lstmp_cell_3/MatMul/ReadVariableOp2L
$lstmp_cell_3/MatMul_1/ReadVariableOp$lstmp_cell_3/MatMul_1/ReadVariableOp2L
$lstmp_cell_3/MatMul_2/ReadVariableOp$lstmp_cell_3/MatMul_2/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
?


while_body_1431
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstmp_cell_3_matmul_readvariableop_resource_0:	@H
5while_lstmp_cell_3_matmul_1_readvariableop_resource_0:	@C
4while_lstmp_cell_3_biasadd_readvariableop_resource_0:	G
5while_lstmp_cell_3_matmul_2_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstmp_cell_3_matmul_readvariableop_resource:	@F
3while_lstmp_cell_3_matmul_1_readvariableop_resource:	@A
2while_lstmp_cell_3_biasadd_readvariableop_resource:	E
3while_lstmp_cell_3_matmul_2_readvariableop_resource:@@¢)while/lstmp_cell_3/BiasAdd/ReadVariableOp¢(while/lstmp_cell_3/MatMul/ReadVariableOp¢*while/lstmp_cell_3/MatMul_1/ReadVariableOp¢*while/lstmp_cell_3/MatMul_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0
(while/lstmp_cell_3/MatMul/ReadVariableOpReadVariableOp3while_lstmp_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	@*
dtype0º
while/lstmp_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstmp_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
*while/lstmp_cell_3/MatMul_1/ReadVariableOpReadVariableOp5while_lstmp_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes
:	@*
dtype0¡
while/lstmp_cell_3/MatMul_1MatMulwhile_placeholder_22while/lstmp_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstmp_cell_3/addAddV2#while/lstmp_cell_3/MatMul:product:0%while/lstmp_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstmp_cell_3/BiasAdd/ReadVariableOpReadVariableOp4while_lstmp_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstmp_cell_3/BiasAddBiasAddwhile/lstmp_cell_3/add:z:01while/lstmp_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstmp_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstmp_cell_3/splitSplit+while/lstmp_cell_3/split/split_dim:output:0#while/lstmp_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*
	num_splitz
while/lstmp_cell_3/SigmoidSigmoid!while/lstmp_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
while/lstmp_cell_3/Sigmoid_1Sigmoid!while/lstmp_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_3/mulMul while/lstmp_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@t
while/lstmp_cell_3/TanhTanh!while/lstmp_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_3/mul_1Mulwhile/lstmp_cell_3/Sigmoid:y:0while/lstmp_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_3/add_1AddV2while/lstmp_cell_3/mul:z:0while/lstmp_cell_3/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
while/lstmp_cell_3/Sigmoid_2Sigmoid!while/lstmp_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
while/lstmp_cell_3/Tanh_1Tanhwhile/lstmp_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_3/mul_2Mul while/lstmp_cell_3/Sigmoid_2:y:0while/lstmp_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
*while/lstmp_cell_3/MatMul_2/ReadVariableOpReadVariableOp5while_lstmp_cell_3_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype0 
while/lstmp_cell_3/MatMul_2MatMulwhile_placeholder_22while/lstmp_cell_3/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : í
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstmp_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity%while/lstmp_cell_3/MatMul_2:product:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/Identity_5Identitywhile/lstmp_cell_3/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ý

while/NoOpNoOp*^while/lstmp_cell_3/BiasAdd/ReadVariableOp)^while/lstmp_cell_3/MatMul/ReadVariableOp+^while/lstmp_cell_3/MatMul_1/ReadVariableOp+^while/lstmp_cell_3/MatMul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstmp_cell_3_biasadd_readvariableop_resource4while_lstmp_cell_3_biasadd_readvariableop_resource_0"l
3while_lstmp_cell_3_matmul_1_readvariableop_resource5while_lstmp_cell_3_matmul_1_readvariableop_resource_0"l
3while_lstmp_cell_3_matmul_2_readvariableop_resource5while_lstmp_cell_3_matmul_2_readvariableop_resource_0"h
1while_lstmp_cell_3_matmul_readvariableop_resource3while_lstmp_cell_3_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : 2V
)while/lstmp_cell_3/BiasAdd/ReadVariableOp)while/lstmp_cell_3/BiasAdd/ReadVariableOp2T
(while/lstmp_cell_3/MatMul/ReadVariableOp(while/lstmp_cell_3/MatMul/ReadVariableOp2X
*while/lstmp_cell_3/MatMul_1/ReadVariableOp*while/lstmp_cell_3/MatMul_1/ReadVariableOp2X
*while/lstmp_cell_3/MatMul_2/ReadVariableOp*while/lstmp_cell_3/MatMul_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
ú=


while_body_1837
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstmp_cell_2_matmul_readvariableop_resource_0:	H
5while_lstmp_cell_2_matmul_1_readvariableop_resource_0:	@C
4while_lstmp_cell_2_biasadd_readvariableop_resource_0:	G
5while_lstmp_cell_2_matmul_2_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstmp_cell_2_matmul_readvariableop_resource:	F
3while_lstmp_cell_2_matmul_1_readvariableop_resource:	@A
2while_lstmp_cell_2_biasadd_readvariableop_resource:	E
3while_lstmp_cell_2_matmul_2_readvariableop_resource:@@¢)while/lstmp_cell_2/BiasAdd/ReadVariableOp¢(while/lstmp_cell_2/MatMul/ReadVariableOp¢*while/lstmp_cell_2/MatMul_1/ReadVariableOp¢*while/lstmp_cell_2/MatMul_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstmp_cell_2/MatMul/ReadVariableOpReadVariableOp3while_lstmp_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstmp_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstmp_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
*while/lstmp_cell_2/MatMul_1/ReadVariableOpReadVariableOp5while_lstmp_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes
:	@*
dtype0¡
while/lstmp_cell_2/MatMul_1MatMulwhile_placeholder_22while/lstmp_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstmp_cell_2/addAddV2#while/lstmp_cell_2/MatMul:product:0%while/lstmp_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstmp_cell_2/BiasAdd/ReadVariableOpReadVariableOp4while_lstmp_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstmp_cell_2/BiasAddBiasAddwhile/lstmp_cell_2/add:z:01while/lstmp_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstmp_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstmp_cell_2/splitSplit+while/lstmp_cell_2/split/split_dim:output:0#while/lstmp_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*
	num_splitz
while/lstmp_cell_2/SigmoidSigmoid!while/lstmp_cell_2/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
while/lstmp_cell_2/Sigmoid_1Sigmoid!while/lstmp_cell_2/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_2/mulMul while/lstmp_cell_2/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@t
while/lstmp_cell_2/TanhTanh!while/lstmp_cell_2/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_2/mul_1Mulwhile/lstmp_cell_2/Sigmoid:y:0while/lstmp_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_2/add_1AddV2while/lstmp_cell_2/mul:z:0while/lstmp_cell_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
while/lstmp_cell_2/Sigmoid_2Sigmoid!while/lstmp_cell_2/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
while/lstmp_cell_2/Tanh_1Tanhwhile/lstmp_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_2/mul_2Mul while/lstmp_cell_2/Sigmoid_2:y:0while/lstmp_cell_2/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
*while/lstmp_cell_2/MatMul_2/ReadVariableOpReadVariableOp5while_lstmp_cell_2_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype0 
while/lstmp_cell_2/MatMul_2MatMulwhile_placeholder_22while/lstmp_cell_2/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstmp_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity%while/lstmp_cell_2/MatMul_2:product:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/Identity_5Identitywhile/lstmp_cell_2/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ý

while/NoOpNoOp*^while/lstmp_cell_2/BiasAdd/ReadVariableOp)^while/lstmp_cell_2/MatMul/ReadVariableOp+^while/lstmp_cell_2/MatMul_1/ReadVariableOp+^while/lstmp_cell_2/MatMul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstmp_cell_2_biasadd_readvariableop_resource4while_lstmp_cell_2_biasadd_readvariableop_resource_0"l
3while_lstmp_cell_2_matmul_1_readvariableop_resource5while_lstmp_cell_2_matmul_1_readvariableop_resource_0"l
3while_lstmp_cell_2_matmul_2_readvariableop_resource5while_lstmp_cell_2_matmul_2_readvariableop_resource_0"h
1while_lstmp_cell_2_matmul_readvariableop_resource3while_lstmp_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : 2V
)while/lstmp_cell_2/BiasAdd/ReadVariableOp)while/lstmp_cell_2/BiasAdd/ReadVariableOp2T
(while/lstmp_cell_2/MatMul/ReadVariableOp(while/lstmp_cell_2/MatMul/ReadVariableOp2X
*while/lstmp_cell_2/MatMul_1/ReadVariableOp*while/lstmp_cell_2/MatMul_1/ReadVariableOp2X
*while/lstmp_cell_2/MatMul_2/ReadVariableOp*while/lstmp_cell_2/MatMul_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
æM

A__inference_lstmp_2_layer_call_and_return_conditional_losses_2965

inputs>
+lstmp_cell_2_matmul_readvariableop_resource:	@
-lstmp_cell_2_matmul_1_readvariableop_resource:	@;
,lstmp_cell_2_biasadd_readvariableop_resource:	?
-lstmp_cell_2_matmul_2_readvariableop_resource:@@
identity¢#lstmp_cell_2/BiasAdd/ReadVariableOp¢"lstmp_cell_2/MatMul/ReadVariableOp¢$lstmp_cell_2/MatMul_1/ReadVariableOp¢$lstmp_cell_2/MatMul_2/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstmp_cell_2/MatMul/ReadVariableOpReadVariableOp+lstmp_cell_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstmp_cell_2/MatMulMatMulstrided_slice_2:output:0*lstmp_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstmp_cell_2/MatMul_1/ReadVariableOpReadVariableOp-lstmp_cell_2_matmul_1_readvariableop_resource*
_output_shapes
:	@*
dtype0
lstmp_cell_2/MatMul_1MatMulzeros:output:0,lstmp_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstmp_cell_2/addAddV2lstmp_cell_2/MatMul:product:0lstmp_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstmp_cell_2/BiasAdd/ReadVariableOpReadVariableOp,lstmp_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstmp_cell_2/BiasAddBiasAddlstmp_cell_2/add:z:0+lstmp_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstmp_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstmp_cell_2/splitSplit%lstmp_cell_2/split/split_dim:output:0lstmp_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*
	num_splitn
lstmp_cell_2/SigmoidSigmoidlstmp_cell_2/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@p
lstmp_cell_2/Sigmoid_1Sigmoidlstmp_cell_2/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
lstmp_cell_2/mulMullstmp_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
lstmp_cell_2/TanhTanhlstmp_cell_2/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
lstmp_cell_2/mul_1Mullstmp_cell_2/Sigmoid:y:0lstmp_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
lstmp_cell_2/add_1AddV2lstmp_cell_2/mul:z:0lstmp_cell_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@p
lstmp_cell_2/Sigmoid_2Sigmoidlstmp_cell_2/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
lstmp_cell_2/Tanh_1Tanhlstmp_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_cell_2/mul_2Mullstmp_cell_2/Sigmoid_2:y:0lstmp_cell_2/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
$lstmp_cell_2/MatMul_2/ReadVariableOpReadVariableOp-lstmp_cell_2_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype0
lstmp_cell_2/MatMul_2MatMulzeros:output:0,lstmp_cell_2/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ³
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstmp_cell_2_matmul_readvariableop_resource-lstmp_cell_2_matmul_1_readvariableop_resource,lstmp_cell_2_biasadd_readvariableop_resource-lstmp_cell_2_matmul_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *&
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_2877*
condR
while_cond_2876*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ç
NoOpNoOp$^lstmp_cell_2/BiasAdd/ReadVariableOp#^lstmp_cell_2/MatMul/ReadVariableOp%^lstmp_cell_2/MatMul_1/ReadVariableOp%^lstmp_cell_2/MatMul_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : 2J
#lstmp_cell_2/BiasAdd/ReadVariableOp#lstmp_cell_2/BiasAdd/ReadVariableOp2H
"lstmp_cell_2/MatMul/ReadVariableOp"lstmp_cell_2/MatMul/ReadVariableOp2L
$lstmp_cell_2/MatMul_1/ReadVariableOp$lstmp_cell_2/MatMul_1/ReadVariableOp2L
$lstmp_cell_2/MatMul_2/ReadVariableOp$lstmp_cell_2/MatMul_2/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
Ð
&__inference_lstmp_2_layer_call_fn_2802

inputs
unknown:	
	unknown_0:	@
	unknown_1:	
	unknown_2:@@
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstmp_2_layer_call_and_return_conditional_losses_1358s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
H
£
lstmp_2_while_body_2547,
(lstmp_2_while_lstmp_2_while_loop_counter2
.lstmp_2_while_lstmp_2_while_maximum_iterations
lstmp_2_while_placeholder
lstmp_2_while_placeholder_1
lstmp_2_while_placeholder_2
lstmp_2_while_placeholder_3+
'lstmp_2_while_lstmp_2_strided_slice_1_0g
clstmp_2_while_tensorarrayv2read_tensorlistgetitem_lstmp_2_tensorarrayunstack_tensorlistfromtensor_0N
;lstmp_2_while_lstmp_cell_2_matmul_readvariableop_resource_0:	P
=lstmp_2_while_lstmp_cell_2_matmul_1_readvariableop_resource_0:	@K
<lstmp_2_while_lstmp_cell_2_biasadd_readvariableop_resource_0:	O
=lstmp_2_while_lstmp_cell_2_matmul_2_readvariableop_resource_0:@@
lstmp_2_while_identity
lstmp_2_while_identity_1
lstmp_2_while_identity_2
lstmp_2_while_identity_3
lstmp_2_while_identity_4
lstmp_2_while_identity_5)
%lstmp_2_while_lstmp_2_strided_slice_1e
alstmp_2_while_tensorarrayv2read_tensorlistgetitem_lstmp_2_tensorarrayunstack_tensorlistfromtensorL
9lstmp_2_while_lstmp_cell_2_matmul_readvariableop_resource:	N
;lstmp_2_while_lstmp_cell_2_matmul_1_readvariableop_resource:	@I
:lstmp_2_while_lstmp_cell_2_biasadd_readvariableop_resource:	M
;lstmp_2_while_lstmp_cell_2_matmul_2_readvariableop_resource:@@¢1lstmp_2/while/lstmp_cell_2/BiasAdd/ReadVariableOp¢0lstmp_2/while/lstmp_cell_2/MatMul/ReadVariableOp¢2lstmp_2/while/lstmp_cell_2/MatMul_1/ReadVariableOp¢2lstmp_2/while/lstmp_cell_2/MatMul_2/ReadVariableOp
?lstmp_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Î
1lstmp_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstmp_2_while_tensorarrayv2read_tensorlistgetitem_lstmp_2_tensorarrayunstack_tensorlistfromtensor_0lstmp_2_while_placeholderHlstmp_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0­
0lstmp_2/while/lstmp_cell_2/MatMul/ReadVariableOpReadVariableOp;lstmp_2_while_lstmp_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ò
!lstmp_2/while/lstmp_cell_2/MatMulMatMul8lstmp_2/while/TensorArrayV2Read/TensorListGetItem:item:08lstmp_2/while/lstmp_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
2lstmp_2/while/lstmp_cell_2/MatMul_1/ReadVariableOpReadVariableOp=lstmp_2_while_lstmp_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes
:	@*
dtype0¹
#lstmp_2/while/lstmp_cell_2/MatMul_1MatMullstmp_2_while_placeholder_2:lstmp_2/while/lstmp_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
lstmp_2/while/lstmp_cell_2/addAddV2+lstmp_2/while/lstmp_cell_2/MatMul:product:0-lstmp_2/while/lstmp_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
1lstmp_2/while/lstmp_cell_2/BiasAdd/ReadVariableOpReadVariableOp<lstmp_2_while_lstmp_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0¿
"lstmp_2/while/lstmp_cell_2/BiasAddBiasAdd"lstmp_2/while/lstmp_cell_2/add:z:09lstmp_2/while/lstmp_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
*lstmp_2/while/lstmp_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstmp_2/while/lstmp_cell_2/splitSplit3lstmp_2/while/lstmp_cell_2/split/split_dim:output:0+lstmp_2/while/lstmp_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*
	num_split
"lstmp_2/while/lstmp_cell_2/SigmoidSigmoid)lstmp_2/while/lstmp_cell_2/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
$lstmp_2/while/lstmp_cell_2/Sigmoid_1Sigmoid)lstmp_2/while/lstmp_cell_2/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_2/while/lstmp_cell_2/mulMul(lstmp_2/while/lstmp_cell_2/Sigmoid_1:y:0lstmp_2_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_2/while/lstmp_cell_2/TanhTanh)lstmp_2/while/lstmp_cell_2/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
 lstmp_2/while/lstmp_cell_2/mul_1Mul&lstmp_2/while/lstmp_cell_2/Sigmoid:y:0#lstmp_2/while/lstmp_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
 lstmp_2/while/lstmp_cell_2/add_1AddV2"lstmp_2/while/lstmp_cell_2/mul:z:0$lstmp_2/while/lstmp_cell_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
$lstmp_2/while/lstmp_cell_2/Sigmoid_2Sigmoid)lstmp_2/while/lstmp_cell_2/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
!lstmp_2/while/lstmp_cell_2/Tanh_1Tanh$lstmp_2/while/lstmp_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ª
 lstmp_2/while/lstmp_cell_2/mul_2Mul(lstmp_2/while/lstmp_cell_2/Sigmoid_2:y:0%lstmp_2/while/lstmp_cell_2/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@°
2lstmp_2/while/lstmp_cell_2/MatMul_2/ReadVariableOpReadVariableOp=lstmp_2_while_lstmp_cell_2_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¸
#lstmp_2/while/lstmp_cell_2/MatMul_2MatMullstmp_2_while_placeholder_2:lstmp_2/while/lstmp_cell_2/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@å
2lstmp_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstmp_2_while_placeholder_1lstmp_2_while_placeholder$lstmp_2/while/lstmp_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒU
lstmp_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstmp_2/while/addAddV2lstmp_2_while_placeholderlstmp_2/while/add/y:output:0*
T0*
_output_shapes
: W
lstmp_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstmp_2/while/add_1AddV2(lstmp_2_while_lstmp_2_while_loop_counterlstmp_2/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstmp_2/while/IdentityIdentitylstmp_2/while/add_1:z:0^lstmp_2/while/NoOp*
T0*
_output_shapes
: 
lstmp_2/while/Identity_1Identity.lstmp_2_while_lstmp_2_while_maximum_iterations^lstmp_2/while/NoOp*
T0*
_output_shapes
: q
lstmp_2/while/Identity_2Identitylstmp_2/while/add:z:0^lstmp_2/while/NoOp*
T0*
_output_shapes
: 
lstmp_2/while/Identity_3IdentityBlstmp_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstmp_2/while/NoOp*
T0*
_output_shapes
: 
lstmp_2/while/Identity_4Identity-lstmp_2/while/lstmp_cell_2/MatMul_2:product:0^lstmp_2/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_2/while/Identity_5Identity$lstmp_2/while/lstmp_cell_2/add_1:z:0^lstmp_2/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
lstmp_2/while/NoOpNoOp2^lstmp_2/while/lstmp_cell_2/BiasAdd/ReadVariableOp1^lstmp_2/while/lstmp_cell_2/MatMul/ReadVariableOp3^lstmp_2/while/lstmp_cell_2/MatMul_1/ReadVariableOp3^lstmp_2/while/lstmp_cell_2/MatMul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstmp_2_while_identitylstmp_2/while/Identity:output:0"=
lstmp_2_while_identity_1!lstmp_2/while/Identity_1:output:0"=
lstmp_2_while_identity_2!lstmp_2/while/Identity_2:output:0"=
lstmp_2_while_identity_3!lstmp_2/while/Identity_3:output:0"=
lstmp_2_while_identity_4!lstmp_2/while/Identity_4:output:0"=
lstmp_2_while_identity_5!lstmp_2/while/Identity_5:output:0"P
%lstmp_2_while_lstmp_2_strided_slice_1'lstmp_2_while_lstmp_2_strided_slice_1_0"z
:lstmp_2_while_lstmp_cell_2_biasadd_readvariableop_resource<lstmp_2_while_lstmp_cell_2_biasadd_readvariableop_resource_0"|
;lstmp_2_while_lstmp_cell_2_matmul_1_readvariableop_resource=lstmp_2_while_lstmp_cell_2_matmul_1_readvariableop_resource_0"|
;lstmp_2_while_lstmp_cell_2_matmul_2_readvariableop_resource=lstmp_2_while_lstmp_cell_2_matmul_2_readvariableop_resource_0"x
9lstmp_2_while_lstmp_cell_2_matmul_readvariableop_resource;lstmp_2_while_lstmp_cell_2_matmul_readvariableop_resource_0"È
alstmp_2_while_tensorarrayv2read_tensorlistgetitem_lstmp_2_tensorarrayunstack_tensorlistfromtensorclstmp_2_while_tensorarrayv2read_tensorlistgetitem_lstmp_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : 2f
1lstmp_2/while/lstmp_cell_2/BiasAdd/ReadVariableOp1lstmp_2/while/lstmp_cell_2/BiasAdd/ReadVariableOp2d
0lstmp_2/while/lstmp_cell_2/MatMul/ReadVariableOp0lstmp_2/while/lstmp_cell_2/MatMul/ReadVariableOp2h
2lstmp_2/while/lstmp_cell_2/MatMul_1/ReadVariableOp2lstmp_2/while/lstmp_cell_2/MatMul_1/ReadVariableOp2h
2lstmp_2/while/lstmp_cell_2/MatMul_2/ReadVariableOp2lstmp_2/while/lstmp_cell_2/MatMul_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
£

@__inference_dlstmp_layer_call_and_return_conditional_losses_2065
input_2
lstmp_2_2041:	
lstmp_2_2043:	@
lstmp_2_2045:	
lstmp_2_2047:@@
lstmp_3_2050:	@
lstmp_3_2052:	@
lstmp_3_2054:	
lstmp_3_2056:@@
dense_1_2059:@
dense_1_2061:
identity¢dense_1/StatefulPartitionedCall¢lstmp_2/StatefulPartitionedCall¢lstmp_3/StatefulPartitionedCall
lstmp_2/StatefulPartitionedCallStatefulPartitionedCallinput_2lstmp_2_2041lstmp_2_2043lstmp_2_2045lstmp_2_2047*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstmp_2_layer_call_and_return_conditional_losses_1358«
lstmp_3/StatefulPartitionedCallStatefulPartitionedCall(lstmp_2/StatefulPartitionedCall:output:0lstmp_3_2050lstmp_3_2052lstmp_3_2054lstmp_3_2056*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstmp_3_layer_call_and_return_conditional_losses_1520
dense_1/StatefulPartitionedCallStatefulPartitionedCall(lstmp_3/StatefulPartitionedCall:output:0dense_1_2059dense_1_2061*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_1540w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
NoOpNoOp ^dense_1/StatefulPartitionedCall ^lstmp_2/StatefulPartitionedCall ^lstmp_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
lstmp_2/StatefulPartitionedCalllstmp_2/StatefulPartitionedCall2B
lstmp_3/StatefulPartitionedCalllstmp_3/StatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
Ô


lstmp_2_while_cond_2242,
(lstmp_2_while_lstmp_2_while_loop_counter2
.lstmp_2_while_lstmp_2_while_maximum_iterations
lstmp_2_while_placeholder
lstmp_2_while_placeholder_1
lstmp_2_while_placeholder_2
lstmp_2_while_placeholder_3.
*lstmp_2_while_less_lstmp_2_strided_slice_1B
>lstmp_2_while_lstmp_2_while_cond_2242___redundant_placeholder0B
>lstmp_2_while_lstmp_2_while_cond_2242___redundant_placeholder1B
>lstmp_2_while_lstmp_2_while_cond_2242___redundant_placeholder2B
>lstmp_2_while_lstmp_2_while_cond_2242___redundant_placeholder3B
>lstmp_2_while_lstmp_2_while_cond_2242___redundant_placeholder4
lstmp_2_while_identity

lstmp_2/while/LessLesslstmp_2_while_placeholder*lstmp_2_while_less_lstmp_2_strided_slice_1*
T0*
_output_shapes
: [
lstmp_2/while/IdentityIdentitylstmp_2/while/Less:z:0*
T0
*
_output_shapes
: "9
lstmp_2_while_identitylstmp_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
ã
í
while_cond_1269
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_12
.while_while_cond_1269___redundant_placeholder02
.while_while_cond_1269___redundant_placeholder12
.while_while_cond_1269___redundant_placeholder22
.while_while_cond_1269___redundant_placeholder32
.while_while_cond_1269___redundant_placeholder4
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
Ô


lstmp_2_while_cond_2546,
(lstmp_2_while_lstmp_2_while_loop_counter2
.lstmp_2_while_lstmp_2_while_maximum_iterations
lstmp_2_while_placeholder
lstmp_2_while_placeholder_1
lstmp_2_while_placeholder_2
lstmp_2_while_placeholder_3.
*lstmp_2_while_less_lstmp_2_strided_slice_1B
>lstmp_2_while_lstmp_2_while_cond_2546___redundant_placeholder0B
>lstmp_2_while_lstmp_2_while_cond_2546___redundant_placeholder1B
>lstmp_2_while_lstmp_2_while_cond_2546___redundant_placeholder2B
>lstmp_2_while_lstmp_2_while_cond_2546___redundant_placeholder3B
>lstmp_2_while_lstmp_2_while_cond_2546___redundant_placeholder4
lstmp_2_while_identity

lstmp_2/while/LessLesslstmp_2_while_placeholder*lstmp_2_while_less_lstmp_2_strided_slice_1*
T0*
_output_shapes
: [
lstmp_2/while/IdentityIdentitylstmp_2/while/Less:z:0*
T0
*
_output_shapes
: "9
lstmp_2_while_identitylstmp_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
ÆÈ
Ý

__inference__wrapped_model_1201
input_2M
:dlstmp_lstmp_2_lstmp_cell_2_matmul_readvariableop_resource:	O
<dlstmp_lstmp_2_lstmp_cell_2_matmul_1_readvariableop_resource:	@J
;dlstmp_lstmp_2_lstmp_cell_2_biasadd_readvariableop_resource:	N
<dlstmp_lstmp_2_lstmp_cell_2_matmul_2_readvariableop_resource:@@M
:dlstmp_lstmp_3_lstmp_cell_3_matmul_readvariableop_resource:	@O
<dlstmp_lstmp_3_lstmp_cell_3_matmul_1_readvariableop_resource:	@J
;dlstmp_lstmp_3_lstmp_cell_3_biasadd_readvariableop_resource:	N
<dlstmp_lstmp_3_lstmp_cell_3_matmul_2_readvariableop_resource:@@?
-dlstmp_dense_1_matmul_readvariableop_resource:@<
.dlstmp_dense_1_biasadd_readvariableop_resource:
identity¢%dlstmp/dense_1/BiasAdd/ReadVariableOp¢$dlstmp/dense_1/MatMul/ReadVariableOp¢2dlstmp/lstmp_2/lstmp_cell_2/BiasAdd/ReadVariableOp¢1dlstmp/lstmp_2/lstmp_cell_2/MatMul/ReadVariableOp¢3dlstmp/lstmp_2/lstmp_cell_2/MatMul_1/ReadVariableOp¢3dlstmp/lstmp_2/lstmp_cell_2/MatMul_2/ReadVariableOp¢dlstmp/lstmp_2/while¢2dlstmp/lstmp_3/lstmp_cell_3/BiasAdd/ReadVariableOp¢1dlstmp/lstmp_3/lstmp_cell_3/MatMul/ReadVariableOp¢3dlstmp/lstmp_3/lstmp_cell_3/MatMul_1/ReadVariableOp¢3dlstmp/lstmp_3/lstmp_cell_3/MatMul_2/ReadVariableOp¢dlstmp/lstmp_3/whileK
dlstmp/lstmp_2/ShapeShapeinput_2*
T0*
_output_shapes
:l
"dlstmp/lstmp_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$dlstmp/lstmp_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$dlstmp/lstmp_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
dlstmp/lstmp_2/strided_sliceStridedSlicedlstmp/lstmp_2/Shape:output:0+dlstmp/lstmp_2/strided_slice/stack:output:0-dlstmp/lstmp_2/strided_slice/stack_1:output:0-dlstmp/lstmp_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
dlstmp/lstmp_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@ 
dlstmp/lstmp_2/zeros/packedPack%dlstmp/lstmp_2/strided_slice:output:0&dlstmp/lstmp_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:_
dlstmp/lstmp_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
dlstmp/lstmp_2/zerosFill$dlstmp/lstmp_2/zeros/packed:output:0#dlstmp/lstmp_2/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
dlstmp/lstmp_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@¤
dlstmp/lstmp_2/zeros_1/packedPack%dlstmp/lstmp_2/strided_slice:output:0(dlstmp/lstmp_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:a
dlstmp/lstmp_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
dlstmp/lstmp_2/zeros_1Fill&dlstmp/lstmp_2/zeros_1/packed:output:0%dlstmp/lstmp_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
dlstmp/lstmp_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
dlstmp/lstmp_2/transpose	Transposeinput_2&dlstmp/lstmp_2/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dlstmp/lstmp_2/Shape_1Shapedlstmp/lstmp_2/transpose:y:0*
T0*
_output_shapes
:n
$dlstmp/lstmp_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&dlstmp/lstmp_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&dlstmp/lstmp_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¦
dlstmp/lstmp_2/strided_slice_1StridedSlicedlstmp/lstmp_2/Shape_1:output:0-dlstmp/lstmp_2/strided_slice_1/stack:output:0/dlstmp/lstmp_2/strided_slice_1/stack_1:output:0/dlstmp/lstmp_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
*dlstmp/lstmp_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿá
dlstmp/lstmp_2/TensorArrayV2TensorListReserve3dlstmp/lstmp_2/TensorArrayV2/element_shape:output:0'dlstmp/lstmp_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Ddlstmp/lstmp_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
6dlstmp/lstmp_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensordlstmp/lstmp_2/transpose:y:0Mdlstmp/lstmp_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒn
$dlstmp/lstmp_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&dlstmp/lstmp_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&dlstmp/lstmp_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:´
dlstmp/lstmp_2/strided_slice_2StridedSlicedlstmp/lstmp_2/transpose:y:0-dlstmp/lstmp_2/strided_slice_2/stack:output:0/dlstmp/lstmp_2/strided_slice_2/stack_1:output:0/dlstmp/lstmp_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask­
1dlstmp/lstmp_2/lstmp_cell_2/MatMul/ReadVariableOpReadVariableOp:dlstmp_lstmp_2_lstmp_cell_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Ã
"dlstmp/lstmp_2/lstmp_cell_2/MatMulMatMul'dlstmp/lstmp_2/strided_slice_2:output:09dlstmp/lstmp_2/lstmp_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
3dlstmp/lstmp_2/lstmp_cell_2/MatMul_1/ReadVariableOpReadVariableOp<dlstmp_lstmp_2_lstmp_cell_2_matmul_1_readvariableop_resource*
_output_shapes
:	@*
dtype0½
$dlstmp/lstmp_2/lstmp_cell_2/MatMul_1MatMuldlstmp/lstmp_2/zeros:output:0;dlstmp/lstmp_2/lstmp_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
dlstmp/lstmp_2/lstmp_cell_2/addAddV2,dlstmp/lstmp_2/lstmp_cell_2/MatMul:product:0.dlstmp/lstmp_2/lstmp_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
2dlstmp/lstmp_2/lstmp_cell_2/BiasAdd/ReadVariableOpReadVariableOp;dlstmp_lstmp_2_lstmp_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Â
#dlstmp/lstmp_2/lstmp_cell_2/BiasAddBiasAdd#dlstmp/lstmp_2/lstmp_cell_2/add:z:0:dlstmp/lstmp_2/lstmp_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
+dlstmp/lstmp_2/lstmp_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
!dlstmp/lstmp_2/lstmp_cell_2/splitSplit4dlstmp/lstmp_2/lstmp_cell_2/split/split_dim:output:0,dlstmp/lstmp_2/lstmp_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*
	num_split
#dlstmp/lstmp_2/lstmp_cell_2/SigmoidSigmoid*dlstmp/lstmp_2/lstmp_cell_2/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
%dlstmp/lstmp_2/lstmp_cell_2/Sigmoid_1Sigmoid*dlstmp/lstmp_2/lstmp_cell_2/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¤
dlstmp/lstmp_2/lstmp_cell_2/mulMul)dlstmp/lstmp_2/lstmp_cell_2/Sigmoid_1:y:0dlstmp/lstmp_2/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 dlstmp/lstmp_2/lstmp_cell_2/TanhTanh*dlstmp/lstmp_2/lstmp_cell_2/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@©
!dlstmp/lstmp_2/lstmp_cell_2/mul_1Mul'dlstmp/lstmp_2/lstmp_cell_2/Sigmoid:y:0$dlstmp/lstmp_2/lstmp_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¨
!dlstmp/lstmp_2/lstmp_cell_2/add_1AddV2#dlstmp/lstmp_2/lstmp_cell_2/mul:z:0%dlstmp/lstmp_2/lstmp_cell_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
%dlstmp/lstmp_2/lstmp_cell_2/Sigmoid_2Sigmoid*dlstmp/lstmp_2/lstmp_cell_2/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"dlstmp/lstmp_2/lstmp_cell_2/Tanh_1Tanh%dlstmp/lstmp_2/lstmp_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@­
!dlstmp/lstmp_2/lstmp_cell_2/mul_2Mul)dlstmp/lstmp_2/lstmp_cell_2/Sigmoid_2:y:0&dlstmp/lstmp_2/lstmp_cell_2/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@°
3dlstmp/lstmp_2/lstmp_cell_2/MatMul_2/ReadVariableOpReadVariableOp<dlstmp_lstmp_2_lstmp_cell_2_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype0¼
$dlstmp/lstmp_2/lstmp_cell_2/MatMul_2MatMuldlstmp/lstmp_2/zeros:output:0;dlstmp/lstmp_2/lstmp_cell_2/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@}
,dlstmp/lstmp_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   å
dlstmp/lstmp_2/TensorArrayV2_1TensorListReserve5dlstmp/lstmp_2/TensorArrayV2_1/element_shape:output:0'dlstmp/lstmp_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒU
dlstmp/lstmp_2/timeConst*
_output_shapes
: *
dtype0*
value	B : r
'dlstmp/lstmp_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿc
!dlstmp/lstmp_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
dlstmp/lstmp_2/whileWhile*dlstmp/lstmp_2/while/loop_counter:output:00dlstmp/lstmp_2/while/maximum_iterations:output:0dlstmp/lstmp_2/time:output:0'dlstmp/lstmp_2/TensorArrayV2_1:handle:0dlstmp/lstmp_2/zeros:output:0dlstmp/lstmp_2/zeros_1:output:0'dlstmp/lstmp_2/strided_slice_1:output:0Fdlstmp/lstmp_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0:dlstmp_lstmp_2_lstmp_cell_2_matmul_readvariableop_resource<dlstmp_lstmp_2_lstmp_cell_2_matmul_1_readvariableop_resource;dlstmp_lstmp_2_lstmp_cell_2_biasadd_readvariableop_resource<dlstmp_lstmp_2_lstmp_cell_2_matmul_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *&
_read_only_resource_inputs
	
*
_stateful_parallelism( *)
body!R
dlstmp_lstmp_2_while_body_959*)
cond!R
dlstmp_lstmp_2_while_cond_958*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *
parallel_iterations 
?dlstmp/lstmp_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ï
1dlstmp/lstmp_2/TensorArrayV2Stack/TensorListStackTensorListStackdlstmp/lstmp_2/while:output:3Hdlstmp/lstmp_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0w
$dlstmp/lstmp_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿp
&dlstmp/lstmp_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: p
&dlstmp/lstmp_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
dlstmp/lstmp_2/strided_slice_3StridedSlice:dlstmp/lstmp_2/TensorArrayV2Stack/TensorListStack:tensor:0-dlstmp/lstmp_2/strided_slice_3/stack:output:0/dlstmp/lstmp_2/strided_slice_3/stack_1:output:0/dlstmp/lstmp_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maskt
dlstmp/lstmp_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ã
dlstmp/lstmp_2/transpose_1	Transpose:dlstmp/lstmp_2/TensorArrayV2Stack/TensorListStack:tensor:0(dlstmp/lstmp_2/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
dlstmp/lstmp_3/ShapeShapedlstmp/lstmp_2/transpose_1:y:0*
T0*
_output_shapes
:l
"dlstmp/lstmp_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$dlstmp/lstmp_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$dlstmp/lstmp_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
dlstmp/lstmp_3/strided_sliceStridedSlicedlstmp/lstmp_3/Shape:output:0+dlstmp/lstmp_3/strided_slice/stack:output:0-dlstmp/lstmp_3/strided_slice/stack_1:output:0-dlstmp/lstmp_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
dlstmp/lstmp_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@ 
dlstmp/lstmp_3/zeros/packedPack%dlstmp/lstmp_3/strided_slice:output:0&dlstmp/lstmp_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:_
dlstmp/lstmp_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
dlstmp/lstmp_3/zerosFill$dlstmp/lstmp_3/zeros/packed:output:0#dlstmp/lstmp_3/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
dlstmp/lstmp_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@¤
dlstmp/lstmp_3/zeros_1/packedPack%dlstmp/lstmp_3/strided_slice:output:0(dlstmp/lstmp_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:a
dlstmp/lstmp_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
dlstmp/lstmp_3/zeros_1Fill&dlstmp/lstmp_3/zeros_1/packed:output:0%dlstmp/lstmp_3/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
dlstmp/lstmp_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          £
dlstmp/lstmp_3/transpose	Transposedlstmp/lstmp_2/transpose_1:y:0&dlstmp/lstmp_3/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
dlstmp/lstmp_3/Shape_1Shapedlstmp/lstmp_3/transpose:y:0*
T0*
_output_shapes
:n
$dlstmp/lstmp_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&dlstmp/lstmp_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&dlstmp/lstmp_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¦
dlstmp/lstmp_3/strided_slice_1StridedSlicedlstmp/lstmp_3/Shape_1:output:0-dlstmp/lstmp_3/strided_slice_1/stack:output:0/dlstmp/lstmp_3/strided_slice_1/stack_1:output:0/dlstmp/lstmp_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
*dlstmp/lstmp_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿá
dlstmp/lstmp_3/TensorArrayV2TensorListReserve3dlstmp/lstmp_3/TensorArrayV2/element_shape:output:0'dlstmp/lstmp_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Ddlstmp/lstmp_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
6dlstmp/lstmp_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensordlstmp/lstmp_3/transpose:y:0Mdlstmp/lstmp_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒn
$dlstmp/lstmp_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&dlstmp/lstmp_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&dlstmp/lstmp_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:´
dlstmp/lstmp_3/strided_slice_2StridedSlicedlstmp/lstmp_3/transpose:y:0-dlstmp/lstmp_3/strided_slice_2/stack:output:0/dlstmp/lstmp_3/strided_slice_2/stack_1:output:0/dlstmp/lstmp_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask­
1dlstmp/lstmp_3/lstmp_cell_3/MatMul/ReadVariableOpReadVariableOp:dlstmp_lstmp_3_lstmp_cell_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0Ã
"dlstmp/lstmp_3/lstmp_cell_3/MatMulMatMul'dlstmp/lstmp_3/strided_slice_2:output:09dlstmp/lstmp_3/lstmp_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
3dlstmp/lstmp_3/lstmp_cell_3/MatMul_1/ReadVariableOpReadVariableOp<dlstmp_lstmp_3_lstmp_cell_3_matmul_1_readvariableop_resource*
_output_shapes
:	@*
dtype0½
$dlstmp/lstmp_3/lstmp_cell_3/MatMul_1MatMuldlstmp/lstmp_3/zeros:output:0;dlstmp/lstmp_3/lstmp_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
dlstmp/lstmp_3/lstmp_cell_3/addAddV2,dlstmp/lstmp_3/lstmp_cell_3/MatMul:product:0.dlstmp/lstmp_3/lstmp_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
2dlstmp/lstmp_3/lstmp_cell_3/BiasAdd/ReadVariableOpReadVariableOp;dlstmp_lstmp_3_lstmp_cell_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Â
#dlstmp/lstmp_3/lstmp_cell_3/BiasAddBiasAdd#dlstmp/lstmp_3/lstmp_cell_3/add:z:0:dlstmp/lstmp_3/lstmp_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
+dlstmp/lstmp_3/lstmp_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
!dlstmp/lstmp_3/lstmp_cell_3/splitSplit4dlstmp/lstmp_3/lstmp_cell_3/split/split_dim:output:0,dlstmp/lstmp_3/lstmp_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*
	num_split
#dlstmp/lstmp_3/lstmp_cell_3/SigmoidSigmoid*dlstmp/lstmp_3/lstmp_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
%dlstmp/lstmp_3/lstmp_cell_3/Sigmoid_1Sigmoid*dlstmp/lstmp_3/lstmp_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¤
dlstmp/lstmp_3/lstmp_cell_3/mulMul)dlstmp/lstmp_3/lstmp_cell_3/Sigmoid_1:y:0dlstmp/lstmp_3/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 dlstmp/lstmp_3/lstmp_cell_3/TanhTanh*dlstmp/lstmp_3/lstmp_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@©
!dlstmp/lstmp_3/lstmp_cell_3/mul_1Mul'dlstmp/lstmp_3/lstmp_cell_3/Sigmoid:y:0$dlstmp/lstmp_3/lstmp_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¨
!dlstmp/lstmp_3/lstmp_cell_3/add_1AddV2#dlstmp/lstmp_3/lstmp_cell_3/mul:z:0%dlstmp/lstmp_3/lstmp_cell_3/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
%dlstmp/lstmp_3/lstmp_cell_3/Sigmoid_2Sigmoid*dlstmp/lstmp_3/lstmp_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"dlstmp/lstmp_3/lstmp_cell_3/Tanh_1Tanh%dlstmp/lstmp_3/lstmp_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@­
!dlstmp/lstmp_3/lstmp_cell_3/mul_2Mul)dlstmp/lstmp_3/lstmp_cell_3/Sigmoid_2:y:0&dlstmp/lstmp_3/lstmp_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@°
3dlstmp/lstmp_3/lstmp_cell_3/MatMul_2/ReadVariableOpReadVariableOp<dlstmp_lstmp_3_lstmp_cell_3_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype0¼
$dlstmp/lstmp_3/lstmp_cell_3/MatMul_2MatMuldlstmp/lstmp_3/zeros:output:0;dlstmp/lstmp_3/lstmp_cell_3/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@}
,dlstmp/lstmp_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   m
+dlstmp/lstmp_3/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :ò
dlstmp/lstmp_3/TensorArrayV2_1TensorListReserve5dlstmp/lstmp_3/TensorArrayV2_1/element_shape:output:04dlstmp/lstmp_3/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒU
dlstmp/lstmp_3/timeConst*
_output_shapes
: *
dtype0*
value	B : r
'dlstmp/lstmp_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿc
!dlstmp/lstmp_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
dlstmp/lstmp_3/whileWhile*dlstmp/lstmp_3/while/loop_counter:output:00dlstmp/lstmp_3/while/maximum_iterations:output:0dlstmp/lstmp_3/time:output:0'dlstmp/lstmp_3/TensorArrayV2_1:handle:0dlstmp/lstmp_3/zeros:output:0dlstmp/lstmp_3/zeros_1:output:0'dlstmp/lstmp_3/strided_slice_1:output:0Fdlstmp/lstmp_3/TensorArrayUnstack/TensorListFromTensor:output_handle:0:dlstmp_lstmp_3_lstmp_cell_3_matmul_readvariableop_resource<dlstmp_lstmp_3_lstmp_cell_3_matmul_1_readvariableop_resource;dlstmp_lstmp_3_lstmp_cell_3_biasadd_readvariableop_resource<dlstmp_lstmp_3_lstmp_cell_3_matmul_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *&
_read_only_resource_inputs
	
*
_stateful_parallelism( **
body"R 
dlstmp_lstmp_3_while_body_1106**
cond"R 
dlstmp_lstmp_3_while_cond_1105*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *
parallel_iterations 
?dlstmp/lstmp_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
1dlstmp/lstmp_3/TensorArrayV2Stack/TensorListStackTensorListStackdlstmp/lstmp_3/while:output:3Hdlstmp/lstmp_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elementsw
$dlstmp/lstmp_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿp
&dlstmp/lstmp_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: p
&dlstmp/lstmp_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
dlstmp/lstmp_3/strided_slice_3StridedSlice:dlstmp/lstmp_3/TensorArrayV2Stack/TensorListStack:tensor:0-dlstmp/lstmp_3/strided_slice_3/stack:output:0/dlstmp/lstmp_3/strided_slice_3/stack_1:output:0/dlstmp/lstmp_3/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maskt
dlstmp/lstmp_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ã
dlstmp/lstmp_3/transpose_1	Transpose:dlstmp/lstmp_3/TensorArrayV2Stack/TensorListStack:tensor:0(dlstmp/lstmp_3/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
$dlstmp/dense_1/MatMul/ReadVariableOpReadVariableOp-dlstmp_dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0¨
dlstmp/dense_1/MatMulMatMul'dlstmp/lstmp_3/strided_slice_3:output:0,dlstmp/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%dlstmp/dense_1/BiasAdd/ReadVariableOpReadVariableOp.dlstmp_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0£
dlstmp/dense_1/BiasAddBiasAdddlstmp/dense_1/MatMul:product:0-dlstmp/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
IdentityIdentitydlstmp/dense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿí
NoOpNoOp&^dlstmp/dense_1/BiasAdd/ReadVariableOp%^dlstmp/dense_1/MatMul/ReadVariableOp3^dlstmp/lstmp_2/lstmp_cell_2/BiasAdd/ReadVariableOp2^dlstmp/lstmp_2/lstmp_cell_2/MatMul/ReadVariableOp4^dlstmp/lstmp_2/lstmp_cell_2/MatMul_1/ReadVariableOp4^dlstmp/lstmp_2/lstmp_cell_2/MatMul_2/ReadVariableOp^dlstmp/lstmp_2/while3^dlstmp/lstmp_3/lstmp_cell_3/BiasAdd/ReadVariableOp2^dlstmp/lstmp_3/lstmp_cell_3/MatMul/ReadVariableOp4^dlstmp/lstmp_3/lstmp_cell_3/MatMul_1/ReadVariableOp4^dlstmp/lstmp_3/lstmp_cell_3/MatMul_2/ReadVariableOp^dlstmp/lstmp_3/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2N
%dlstmp/dense_1/BiasAdd/ReadVariableOp%dlstmp/dense_1/BiasAdd/ReadVariableOp2L
$dlstmp/dense_1/MatMul/ReadVariableOp$dlstmp/dense_1/MatMul/ReadVariableOp2h
2dlstmp/lstmp_2/lstmp_cell_2/BiasAdd/ReadVariableOp2dlstmp/lstmp_2/lstmp_cell_2/BiasAdd/ReadVariableOp2f
1dlstmp/lstmp_2/lstmp_cell_2/MatMul/ReadVariableOp1dlstmp/lstmp_2/lstmp_cell_2/MatMul/ReadVariableOp2j
3dlstmp/lstmp_2/lstmp_cell_2/MatMul_1/ReadVariableOp3dlstmp/lstmp_2/lstmp_cell_2/MatMul_1/ReadVariableOp2j
3dlstmp/lstmp_2/lstmp_cell_2/MatMul_2/ReadVariableOp3dlstmp/lstmp_2/lstmp_cell_2/MatMul_2/ReadVariableOp2,
dlstmp/lstmp_2/whiledlstmp/lstmp_2/while2h
2dlstmp/lstmp_3/lstmp_cell_3/BiasAdd/ReadVariableOp2dlstmp/lstmp_3/lstmp_cell_3/BiasAdd/ReadVariableOp2f
1dlstmp/lstmp_3/lstmp_cell_3/MatMul/ReadVariableOp1dlstmp/lstmp_3/lstmp_cell_3/MatMul/ReadVariableOp2j
3dlstmp/lstmp_3/lstmp_cell_3/MatMul_1/ReadVariableOp3dlstmp/lstmp_3/lstmp_cell_3/MatMul_1/ReadVariableOp2j
3dlstmp/lstmp_3/lstmp_cell_3/MatMul_2/ReadVariableOp3dlstmp/lstmp_3/lstmp_cell_3/MatMul_2/ReadVariableOp2,
dlstmp/lstmp_3/whiledlstmp/lstmp_3/while:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
ã
í
while_cond_3026
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_12
.while_while_cond_3026___redundant_placeholder02
.while_while_cond_3026___redundant_placeholder12
.while_while_cond_3026___redundant_placeholder22
.while_while_cond_3026___redundant_placeholder32
.while_while_cond_3026___redundant_placeholder4
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
ã
í
while_cond_1836
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_12
.while_while_cond_1836___redundant_placeholder02
.while_while_cond_1836___redundant_placeholder12
.while_while_cond_1836___redundant_placeholder22
.while_while_cond_1836___redundant_placeholder32
.while_while_cond_1836___redundant_placeholder4
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
£

ø
%__inference_dlstmp_layer_call_fn_2181

inputs
unknown:	
	unknown_0:	@
	unknown_1:	
	unknown_2:@@
	unknown_3:	@
	unknown_4:	@
	unknown_5:	
	unknown_6:@@
	unknown_7:@
	unknown_8:
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dlstmp_layer_call_and_return_conditional_losses_1990o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô


lstmp_3_while_cond_2389,
(lstmp_3_while_lstmp_3_while_loop_counter2
.lstmp_3_while_lstmp_3_while_maximum_iterations
lstmp_3_while_placeholder
lstmp_3_while_placeholder_1
lstmp_3_while_placeholder_2
lstmp_3_while_placeholder_3.
*lstmp_3_while_less_lstmp_3_strided_slice_1B
>lstmp_3_while_lstmp_3_while_cond_2389___redundant_placeholder0B
>lstmp_3_while_lstmp_3_while_cond_2389___redundant_placeholder1B
>lstmp_3_while_lstmp_3_while_cond_2389___redundant_placeholder2B
>lstmp_3_while_lstmp_3_while_cond_2389___redundant_placeholder3B
>lstmp_3_while_lstmp_3_while_cond_2389___redundant_placeholder4
lstmp_3_while_identity

lstmp_3/while/LessLesslstmp_3_while_placeholder*lstmp_3_while_less_lstmp_3_strided_slice_1*
T0*
_output_shapes
: [
lstmp_3/while/IdentityIdentitylstmp_3/while/Less:z:0*
T0
*
_output_shapes
: "9
lstmp_3_while_identitylstmp_3/while/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
?


while_body_1659
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstmp_cell_3_matmul_readvariableop_resource_0:	@H
5while_lstmp_cell_3_matmul_1_readvariableop_resource_0:	@C
4while_lstmp_cell_3_biasadd_readvariableop_resource_0:	G
5while_lstmp_cell_3_matmul_2_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstmp_cell_3_matmul_readvariableop_resource:	@F
3while_lstmp_cell_3_matmul_1_readvariableop_resource:	@A
2while_lstmp_cell_3_biasadd_readvariableop_resource:	E
3while_lstmp_cell_3_matmul_2_readvariableop_resource:@@¢)while/lstmp_cell_3/BiasAdd/ReadVariableOp¢(while/lstmp_cell_3/MatMul/ReadVariableOp¢*while/lstmp_cell_3/MatMul_1/ReadVariableOp¢*while/lstmp_cell_3/MatMul_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0
(while/lstmp_cell_3/MatMul/ReadVariableOpReadVariableOp3while_lstmp_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	@*
dtype0º
while/lstmp_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstmp_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
*while/lstmp_cell_3/MatMul_1/ReadVariableOpReadVariableOp5while_lstmp_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes
:	@*
dtype0¡
while/lstmp_cell_3/MatMul_1MatMulwhile_placeholder_22while/lstmp_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstmp_cell_3/addAddV2#while/lstmp_cell_3/MatMul:product:0%while/lstmp_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstmp_cell_3/BiasAdd/ReadVariableOpReadVariableOp4while_lstmp_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstmp_cell_3/BiasAddBiasAddwhile/lstmp_cell_3/add:z:01while/lstmp_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstmp_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstmp_cell_3/splitSplit+while/lstmp_cell_3/split/split_dim:output:0#while/lstmp_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*
	num_splitz
while/lstmp_cell_3/SigmoidSigmoid!while/lstmp_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
while/lstmp_cell_3/Sigmoid_1Sigmoid!while/lstmp_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_3/mulMul while/lstmp_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@t
while/lstmp_cell_3/TanhTanh!while/lstmp_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_3/mul_1Mulwhile/lstmp_cell_3/Sigmoid:y:0while/lstmp_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_3/add_1AddV2while/lstmp_cell_3/mul:z:0while/lstmp_cell_3/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
while/lstmp_cell_3/Sigmoid_2Sigmoid!while/lstmp_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
while/lstmp_cell_3/Tanh_1Tanhwhile/lstmp_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_3/mul_2Mul while/lstmp_cell_3/Sigmoid_2:y:0while/lstmp_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
*while/lstmp_cell_3/MatMul_2/ReadVariableOpReadVariableOp5while_lstmp_cell_3_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype0 
while/lstmp_cell_3/MatMul_2MatMulwhile_placeholder_22while/lstmp_cell_3/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : í
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstmp_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity%while/lstmp_cell_3/MatMul_2:product:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/Identity_5Identitywhile/lstmp_cell_3/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ý

while/NoOpNoOp*^while/lstmp_cell_3/BiasAdd/ReadVariableOp)^while/lstmp_cell_3/MatMul/ReadVariableOp+^while/lstmp_cell_3/MatMul_1/ReadVariableOp+^while/lstmp_cell_3/MatMul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstmp_cell_3_biasadd_readvariableop_resource4while_lstmp_cell_3_biasadd_readvariableop_resource_0"l
3while_lstmp_cell_3_matmul_1_readvariableop_resource5while_lstmp_cell_3_matmul_1_readvariableop_resource_0"l
3while_lstmp_cell_3_matmul_2_readvariableop_resource5while_lstmp_cell_3_matmul_2_readvariableop_resource_0"h
1while_lstmp_cell_3_matmul_readvariableop_resource3while_lstmp_cell_3_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : 2V
)while/lstmp_cell_3/BiasAdd/ReadVariableOp)while/lstmp_cell_3/BiasAdd/ReadVariableOp2T
(while/lstmp_cell_3/MatMul/ReadVariableOp(while/lstmp_cell_3/MatMul/ReadVariableOp2X
*while/lstmp_cell_3/MatMul_1/ReadVariableOp*while/lstmp_cell_3/MatMul_1/ReadVariableOp2X
*while/lstmp_cell_3/MatMul_2/ReadVariableOp*while/lstmp_cell_3/MatMul_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
?


while_body_3206
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstmp_cell_3_matmul_readvariableop_resource_0:	@H
5while_lstmp_cell_3_matmul_1_readvariableop_resource_0:	@C
4while_lstmp_cell_3_biasadd_readvariableop_resource_0:	G
5while_lstmp_cell_3_matmul_2_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstmp_cell_3_matmul_readvariableop_resource:	@F
3while_lstmp_cell_3_matmul_1_readvariableop_resource:	@A
2while_lstmp_cell_3_biasadd_readvariableop_resource:	E
3while_lstmp_cell_3_matmul_2_readvariableop_resource:@@¢)while/lstmp_cell_3/BiasAdd/ReadVariableOp¢(while/lstmp_cell_3/MatMul/ReadVariableOp¢*while/lstmp_cell_3/MatMul_1/ReadVariableOp¢*while/lstmp_cell_3/MatMul_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0
(while/lstmp_cell_3/MatMul/ReadVariableOpReadVariableOp3while_lstmp_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	@*
dtype0º
while/lstmp_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstmp_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
*while/lstmp_cell_3/MatMul_1/ReadVariableOpReadVariableOp5while_lstmp_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes
:	@*
dtype0¡
while/lstmp_cell_3/MatMul_1MatMulwhile_placeholder_22while/lstmp_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstmp_cell_3/addAddV2#while/lstmp_cell_3/MatMul:product:0%while/lstmp_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstmp_cell_3/BiasAdd/ReadVariableOpReadVariableOp4while_lstmp_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstmp_cell_3/BiasAddBiasAddwhile/lstmp_cell_3/add:z:01while/lstmp_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstmp_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstmp_cell_3/splitSplit+while/lstmp_cell_3/split/split_dim:output:0#while/lstmp_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*
	num_splitz
while/lstmp_cell_3/SigmoidSigmoid!while/lstmp_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
while/lstmp_cell_3/Sigmoid_1Sigmoid!while/lstmp_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_3/mulMul while/lstmp_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@t
while/lstmp_cell_3/TanhTanh!while/lstmp_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_3/mul_1Mulwhile/lstmp_cell_3/Sigmoid:y:0while/lstmp_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_3/add_1AddV2while/lstmp_cell_3/mul:z:0while/lstmp_cell_3/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
while/lstmp_cell_3/Sigmoid_2Sigmoid!while/lstmp_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
while/lstmp_cell_3/Tanh_1Tanhwhile/lstmp_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_3/mul_2Mul while/lstmp_cell_3/Sigmoid_2:y:0while/lstmp_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
*while/lstmp_cell_3/MatMul_2/ReadVariableOpReadVariableOp5while_lstmp_cell_3_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype0 
while/lstmp_cell_3/MatMul_2MatMulwhile_placeholder_22while/lstmp_cell_3/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : í
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstmp_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity%while/lstmp_cell_3/MatMul_2:product:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/Identity_5Identitywhile/lstmp_cell_3/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ý

while/NoOpNoOp*^while/lstmp_cell_3/BiasAdd/ReadVariableOp)^while/lstmp_cell_3/MatMul/ReadVariableOp+^while/lstmp_cell_3/MatMul_1/ReadVariableOp+^while/lstmp_cell_3/MatMul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstmp_cell_3_biasadd_readvariableop_resource4while_lstmp_cell_3_biasadd_readvariableop_resource_0"l
3while_lstmp_cell_3_matmul_1_readvariableop_resource5while_lstmp_cell_3_matmul_1_readvariableop_resource_0"l
3while_lstmp_cell_3_matmul_2_readvariableop_resource5while_lstmp_cell_3_matmul_2_readvariableop_resource_0"h
1while_lstmp_cell_3_matmul_readvariableop_resource3while_lstmp_cell_3_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : 2V
)while/lstmp_cell_3/BiasAdd/ReadVariableOp)while/lstmp_cell_3/BiasAdd/ReadVariableOp2T
(while/lstmp_cell_3/MatMul/ReadVariableOp(while/lstmp_cell_3/MatMul/ReadVariableOp2X
*while/lstmp_cell_3/MatMul_1/ReadVariableOp*while/lstmp_cell_3/MatMul_1/ReadVariableOp2X
*while/lstmp_cell_3/MatMul_2/ReadVariableOp*while/lstmp_cell_3/MatMul_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
Q

dlstmp_lstmp_2_while_body_959:
6dlstmp_lstmp_2_while_dlstmp_lstmp_2_while_loop_counter@
<dlstmp_lstmp_2_while_dlstmp_lstmp_2_while_maximum_iterations$
 dlstmp_lstmp_2_while_placeholder&
"dlstmp_lstmp_2_while_placeholder_1&
"dlstmp_lstmp_2_while_placeholder_2&
"dlstmp_lstmp_2_while_placeholder_39
5dlstmp_lstmp_2_while_dlstmp_lstmp_2_strided_slice_1_0u
qdlstmp_lstmp_2_while_tensorarrayv2read_tensorlistgetitem_dlstmp_lstmp_2_tensorarrayunstack_tensorlistfromtensor_0U
Bdlstmp_lstmp_2_while_lstmp_cell_2_matmul_readvariableop_resource_0:	W
Ddlstmp_lstmp_2_while_lstmp_cell_2_matmul_1_readvariableop_resource_0:	@R
Cdlstmp_lstmp_2_while_lstmp_cell_2_biasadd_readvariableop_resource_0:	V
Ddlstmp_lstmp_2_while_lstmp_cell_2_matmul_2_readvariableop_resource_0:@@!
dlstmp_lstmp_2_while_identity#
dlstmp_lstmp_2_while_identity_1#
dlstmp_lstmp_2_while_identity_2#
dlstmp_lstmp_2_while_identity_3#
dlstmp_lstmp_2_while_identity_4#
dlstmp_lstmp_2_while_identity_57
3dlstmp_lstmp_2_while_dlstmp_lstmp_2_strided_slice_1s
odlstmp_lstmp_2_while_tensorarrayv2read_tensorlistgetitem_dlstmp_lstmp_2_tensorarrayunstack_tensorlistfromtensorS
@dlstmp_lstmp_2_while_lstmp_cell_2_matmul_readvariableop_resource:	U
Bdlstmp_lstmp_2_while_lstmp_cell_2_matmul_1_readvariableop_resource:	@P
Adlstmp_lstmp_2_while_lstmp_cell_2_biasadd_readvariableop_resource:	T
Bdlstmp_lstmp_2_while_lstmp_cell_2_matmul_2_readvariableop_resource:@@¢8dlstmp/lstmp_2/while/lstmp_cell_2/BiasAdd/ReadVariableOp¢7dlstmp/lstmp_2/while/lstmp_cell_2/MatMul/ReadVariableOp¢9dlstmp/lstmp_2/while/lstmp_cell_2/MatMul_1/ReadVariableOp¢9dlstmp/lstmp_2/while/lstmp_cell_2/MatMul_2/ReadVariableOp
Fdlstmp/lstmp_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ñ
8dlstmp/lstmp_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqdlstmp_lstmp_2_while_tensorarrayv2read_tensorlistgetitem_dlstmp_lstmp_2_tensorarrayunstack_tensorlistfromtensor_0 dlstmp_lstmp_2_while_placeholderOdlstmp/lstmp_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0»
7dlstmp/lstmp_2/while/lstmp_cell_2/MatMul/ReadVariableOpReadVariableOpBdlstmp_lstmp_2_while_lstmp_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0ç
(dlstmp/lstmp_2/while/lstmp_cell_2/MatMulMatMul?dlstmp/lstmp_2/while/TensorArrayV2Read/TensorListGetItem:item:0?dlstmp/lstmp_2/while/lstmp_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
9dlstmp/lstmp_2/while/lstmp_cell_2/MatMul_1/ReadVariableOpReadVariableOpDdlstmp_lstmp_2_while_lstmp_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes
:	@*
dtype0Î
*dlstmp/lstmp_2/while/lstmp_cell_2/MatMul_1MatMul"dlstmp_lstmp_2_while_placeholder_2Adlstmp/lstmp_2/while/lstmp_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿË
%dlstmp/lstmp_2/while/lstmp_cell_2/addAddV22dlstmp/lstmp_2/while/lstmp_cell_2/MatMul:product:04dlstmp/lstmp_2/while/lstmp_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
8dlstmp/lstmp_2/while/lstmp_cell_2/BiasAdd/ReadVariableOpReadVariableOpCdlstmp_lstmp_2_while_lstmp_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Ô
)dlstmp/lstmp_2/while/lstmp_cell_2/BiasAddBiasAdd)dlstmp/lstmp_2/while/lstmp_cell_2/add:z:0@dlstmp/lstmp_2/while/lstmp_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
1dlstmp/lstmp_2/while/lstmp_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
'dlstmp/lstmp_2/while/lstmp_cell_2/splitSplit:dlstmp/lstmp_2/while/lstmp_cell_2/split/split_dim:output:02dlstmp/lstmp_2/while/lstmp_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*
	num_split
)dlstmp/lstmp_2/while/lstmp_cell_2/SigmoidSigmoid0dlstmp/lstmp_2/while/lstmp_cell_2/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
+dlstmp/lstmp_2/while/lstmp_cell_2/Sigmoid_1Sigmoid0dlstmp/lstmp_2/while/lstmp_cell_2/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@³
%dlstmp/lstmp_2/while/lstmp_cell_2/mulMul/dlstmp/lstmp_2/while/lstmp_cell_2/Sigmoid_1:y:0"dlstmp_lstmp_2_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
&dlstmp/lstmp_2/while/lstmp_cell_2/TanhTanh0dlstmp/lstmp_2/while/lstmp_cell_2/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@»
'dlstmp/lstmp_2/while/lstmp_cell_2/mul_1Mul-dlstmp/lstmp_2/while/lstmp_cell_2/Sigmoid:y:0*dlstmp/lstmp_2/while/lstmp_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
'dlstmp/lstmp_2/while/lstmp_cell_2/add_1AddV2)dlstmp/lstmp_2/while/lstmp_cell_2/mul:z:0+dlstmp/lstmp_2/while/lstmp_cell_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
+dlstmp/lstmp_2/while/lstmp_cell_2/Sigmoid_2Sigmoid0dlstmp/lstmp_2/while/lstmp_cell_2/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(dlstmp/lstmp_2/while/lstmp_cell_2/Tanh_1Tanh+dlstmp/lstmp_2/while/lstmp_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¿
'dlstmp/lstmp_2/while/lstmp_cell_2/mul_2Mul/dlstmp/lstmp_2/while/lstmp_cell_2/Sigmoid_2:y:0,dlstmp/lstmp_2/while/lstmp_cell_2/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¾
9dlstmp/lstmp_2/while/lstmp_cell_2/MatMul_2/ReadVariableOpReadVariableOpDdlstmp_lstmp_2_while_lstmp_cell_2_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype0Í
*dlstmp/lstmp_2/while/lstmp_cell_2/MatMul_2MatMul"dlstmp_lstmp_2_while_placeholder_2Adlstmp/lstmp_2/while/lstmp_cell_2/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
9dlstmp/lstmp_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"dlstmp_lstmp_2_while_placeholder_1 dlstmp_lstmp_2_while_placeholder+dlstmp/lstmp_2/while/lstmp_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒ\
dlstmp/lstmp_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
dlstmp/lstmp_2/while/addAddV2 dlstmp_lstmp_2_while_placeholder#dlstmp/lstmp_2/while/add/y:output:0*
T0*
_output_shapes
: ^
dlstmp/lstmp_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :£
dlstmp/lstmp_2/while/add_1AddV26dlstmp_lstmp_2_while_dlstmp_lstmp_2_while_loop_counter%dlstmp/lstmp_2/while/add_1/y:output:0*
T0*
_output_shapes
: 
dlstmp/lstmp_2/while/IdentityIdentitydlstmp/lstmp_2/while/add_1:z:0^dlstmp/lstmp_2/while/NoOp*
T0*
_output_shapes
: ¦
dlstmp/lstmp_2/while/Identity_1Identity<dlstmp_lstmp_2_while_dlstmp_lstmp_2_while_maximum_iterations^dlstmp/lstmp_2/while/NoOp*
T0*
_output_shapes
: 
dlstmp/lstmp_2/while/Identity_2Identitydlstmp/lstmp_2/while/add:z:0^dlstmp/lstmp_2/while/NoOp*
T0*
_output_shapes
: ³
dlstmp/lstmp_2/while/Identity_3IdentityIdlstmp/lstmp_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^dlstmp/lstmp_2/while/NoOp*
T0*
_output_shapes
: ¯
dlstmp/lstmp_2/while/Identity_4Identity4dlstmp/lstmp_2/while/lstmp_cell_2/MatMul_2:product:0^dlstmp/lstmp_2/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
dlstmp/lstmp_2/while/Identity_5Identity+dlstmp/lstmp_2/while/lstmp_cell_2/add_1:z:0^dlstmp/lstmp_2/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@È
dlstmp/lstmp_2/while/NoOpNoOp9^dlstmp/lstmp_2/while/lstmp_cell_2/BiasAdd/ReadVariableOp8^dlstmp/lstmp_2/while/lstmp_cell_2/MatMul/ReadVariableOp:^dlstmp/lstmp_2/while/lstmp_cell_2/MatMul_1/ReadVariableOp:^dlstmp/lstmp_2/while/lstmp_cell_2/MatMul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "l
3dlstmp_lstmp_2_while_dlstmp_lstmp_2_strided_slice_15dlstmp_lstmp_2_while_dlstmp_lstmp_2_strided_slice_1_0"G
dlstmp_lstmp_2_while_identity&dlstmp/lstmp_2/while/Identity:output:0"K
dlstmp_lstmp_2_while_identity_1(dlstmp/lstmp_2/while/Identity_1:output:0"K
dlstmp_lstmp_2_while_identity_2(dlstmp/lstmp_2/while/Identity_2:output:0"K
dlstmp_lstmp_2_while_identity_3(dlstmp/lstmp_2/while/Identity_3:output:0"K
dlstmp_lstmp_2_while_identity_4(dlstmp/lstmp_2/while/Identity_4:output:0"K
dlstmp_lstmp_2_while_identity_5(dlstmp/lstmp_2/while/Identity_5:output:0"
Adlstmp_lstmp_2_while_lstmp_cell_2_biasadd_readvariableop_resourceCdlstmp_lstmp_2_while_lstmp_cell_2_biasadd_readvariableop_resource_0"
Bdlstmp_lstmp_2_while_lstmp_cell_2_matmul_1_readvariableop_resourceDdlstmp_lstmp_2_while_lstmp_cell_2_matmul_1_readvariableop_resource_0"
Bdlstmp_lstmp_2_while_lstmp_cell_2_matmul_2_readvariableop_resourceDdlstmp_lstmp_2_while_lstmp_cell_2_matmul_2_readvariableop_resource_0"
@dlstmp_lstmp_2_while_lstmp_cell_2_matmul_readvariableop_resourceBdlstmp_lstmp_2_while_lstmp_cell_2_matmul_readvariableop_resource_0"ä
odlstmp_lstmp_2_while_tensorarrayv2read_tensorlistgetitem_dlstmp_lstmp_2_tensorarrayunstack_tensorlistfromtensorqdlstmp_lstmp_2_while_tensorarrayv2read_tensorlistgetitem_dlstmp_lstmp_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : 2t
8dlstmp/lstmp_2/while/lstmp_cell_2/BiasAdd/ReadVariableOp8dlstmp/lstmp_2/while/lstmp_cell_2/BiasAdd/ReadVariableOp2r
7dlstmp/lstmp_2/while/lstmp_cell_2/MatMul/ReadVariableOp7dlstmp/lstmp_2/while/lstmp_cell_2/MatMul/ReadVariableOp2v
9dlstmp/lstmp_2/while/lstmp_cell_2/MatMul_1/ReadVariableOp9dlstmp/lstmp_2/while/lstmp_cell_2/MatMul_1/ReadVariableOp2v
9dlstmp/lstmp_2/while/lstmp_cell_2/MatMul_2/ReadVariableOp9dlstmp/lstmp_2/while/lstmp_cell_2/MatMul_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
ú=


while_body_1270
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstmp_cell_2_matmul_readvariableop_resource_0:	H
5while_lstmp_cell_2_matmul_1_readvariableop_resource_0:	@C
4while_lstmp_cell_2_biasadd_readvariableop_resource_0:	G
5while_lstmp_cell_2_matmul_2_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstmp_cell_2_matmul_readvariableop_resource:	F
3while_lstmp_cell_2_matmul_1_readvariableop_resource:	@A
2while_lstmp_cell_2_biasadd_readvariableop_resource:	E
3while_lstmp_cell_2_matmul_2_readvariableop_resource:@@¢)while/lstmp_cell_2/BiasAdd/ReadVariableOp¢(while/lstmp_cell_2/MatMul/ReadVariableOp¢*while/lstmp_cell_2/MatMul_1/ReadVariableOp¢*while/lstmp_cell_2/MatMul_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstmp_cell_2/MatMul/ReadVariableOpReadVariableOp3while_lstmp_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstmp_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstmp_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
*while/lstmp_cell_2/MatMul_1/ReadVariableOpReadVariableOp5while_lstmp_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes
:	@*
dtype0¡
while/lstmp_cell_2/MatMul_1MatMulwhile_placeholder_22while/lstmp_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstmp_cell_2/addAddV2#while/lstmp_cell_2/MatMul:product:0%while/lstmp_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstmp_cell_2/BiasAdd/ReadVariableOpReadVariableOp4while_lstmp_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstmp_cell_2/BiasAddBiasAddwhile/lstmp_cell_2/add:z:01while/lstmp_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstmp_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstmp_cell_2/splitSplit+while/lstmp_cell_2/split/split_dim:output:0#while/lstmp_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*
	num_splitz
while/lstmp_cell_2/SigmoidSigmoid!while/lstmp_cell_2/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
while/lstmp_cell_2/Sigmoid_1Sigmoid!while/lstmp_cell_2/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_2/mulMul while/lstmp_cell_2/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@t
while/lstmp_cell_2/TanhTanh!while/lstmp_cell_2/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_2/mul_1Mulwhile/lstmp_cell_2/Sigmoid:y:0while/lstmp_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_2/add_1AddV2while/lstmp_cell_2/mul:z:0while/lstmp_cell_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
while/lstmp_cell_2/Sigmoid_2Sigmoid!while/lstmp_cell_2/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
while/lstmp_cell_2/Tanh_1Tanhwhile/lstmp_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_2/mul_2Mul while/lstmp_cell_2/Sigmoid_2:y:0while/lstmp_cell_2/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
*while/lstmp_cell_2/MatMul_2/ReadVariableOpReadVariableOp5while_lstmp_cell_2_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype0 
while/lstmp_cell_2/MatMul_2MatMulwhile_placeholder_22while/lstmp_cell_2/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstmp_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity%while/lstmp_cell_2/MatMul_2:product:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/Identity_5Identitywhile/lstmp_cell_2/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ý

while/NoOpNoOp*^while/lstmp_cell_2/BiasAdd/ReadVariableOp)^while/lstmp_cell_2/MatMul/ReadVariableOp+^while/lstmp_cell_2/MatMul_1/ReadVariableOp+^while/lstmp_cell_2/MatMul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstmp_cell_2_biasadd_readvariableop_resource4while_lstmp_cell_2_biasadd_readvariableop_resource_0"l
3while_lstmp_cell_2_matmul_1_readvariableop_resource5while_lstmp_cell_2_matmul_1_readvariableop_resource_0"l
3while_lstmp_cell_2_matmul_2_readvariableop_resource5while_lstmp_cell_2_matmul_2_readvariableop_resource_0"h
1while_lstmp_cell_2_matmul_readvariableop_resource3while_lstmp_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : 2V
)while/lstmp_cell_2/BiasAdd/ReadVariableOp)while/lstmp_cell_2/BiasAdd/ReadVariableOp2T
(while/lstmp_cell_2/MatMul/ReadVariableOp(while/lstmp_cell_2/MatMul/ReadVariableOp2X
*while/lstmp_cell_2/MatMul_1/ReadVariableOp*while/lstmp_cell_2/MatMul_1/ReadVariableOp2X
*while/lstmp_cell_2/MatMul_2/ReadVariableOp*while/lstmp_cell_2/MatMul_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
ã
í
while_cond_3357
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_12
.while_while_cond_3357___redundant_placeholder02
.while_while_cond_3357___redundant_placeholder12
.while_while_cond_3357___redundant_placeholder22
.while_while_cond_3357___redundant_placeholder32
.while_while_cond_3357___redundant_placeholder4
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
ú=


while_body_3027
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstmp_cell_2_matmul_readvariableop_resource_0:	H
5while_lstmp_cell_2_matmul_1_readvariableop_resource_0:	@C
4while_lstmp_cell_2_biasadd_readvariableop_resource_0:	G
5while_lstmp_cell_2_matmul_2_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstmp_cell_2_matmul_readvariableop_resource:	F
3while_lstmp_cell_2_matmul_1_readvariableop_resource:	@A
2while_lstmp_cell_2_biasadd_readvariableop_resource:	E
3while_lstmp_cell_2_matmul_2_readvariableop_resource:@@¢)while/lstmp_cell_2/BiasAdd/ReadVariableOp¢(while/lstmp_cell_2/MatMul/ReadVariableOp¢*while/lstmp_cell_2/MatMul_1/ReadVariableOp¢*while/lstmp_cell_2/MatMul_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstmp_cell_2/MatMul/ReadVariableOpReadVariableOp3while_lstmp_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstmp_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstmp_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
*while/lstmp_cell_2/MatMul_1/ReadVariableOpReadVariableOp5while_lstmp_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes
:	@*
dtype0¡
while/lstmp_cell_2/MatMul_1MatMulwhile_placeholder_22while/lstmp_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstmp_cell_2/addAddV2#while/lstmp_cell_2/MatMul:product:0%while/lstmp_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstmp_cell_2/BiasAdd/ReadVariableOpReadVariableOp4while_lstmp_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstmp_cell_2/BiasAddBiasAddwhile/lstmp_cell_2/add:z:01while/lstmp_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstmp_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstmp_cell_2/splitSplit+while/lstmp_cell_2/split/split_dim:output:0#while/lstmp_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*
	num_splitz
while/lstmp_cell_2/SigmoidSigmoid!while/lstmp_cell_2/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
while/lstmp_cell_2/Sigmoid_1Sigmoid!while/lstmp_cell_2/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_2/mulMul while/lstmp_cell_2/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@t
while/lstmp_cell_2/TanhTanh!while/lstmp_cell_2/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_2/mul_1Mulwhile/lstmp_cell_2/Sigmoid:y:0while/lstmp_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_2/add_1AddV2while/lstmp_cell_2/mul:z:0while/lstmp_cell_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
while/lstmp_cell_2/Sigmoid_2Sigmoid!while/lstmp_cell_2/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
while/lstmp_cell_2/Tanh_1Tanhwhile/lstmp_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_2/mul_2Mul while/lstmp_cell_2/Sigmoid_2:y:0while/lstmp_cell_2/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
*while/lstmp_cell_2/MatMul_2/ReadVariableOpReadVariableOp5while_lstmp_cell_2_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype0 
while/lstmp_cell_2/MatMul_2MatMulwhile_placeholder_22while/lstmp_cell_2/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstmp_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity%while/lstmp_cell_2/MatMul_2:product:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/Identity_5Identitywhile/lstmp_cell_2/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ý

while/NoOpNoOp*^while/lstmp_cell_2/BiasAdd/ReadVariableOp)^while/lstmp_cell_2/MatMul/ReadVariableOp+^while/lstmp_cell_2/MatMul_1/ReadVariableOp+^while/lstmp_cell_2/MatMul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstmp_cell_2_biasadd_readvariableop_resource4while_lstmp_cell_2_biasadd_readvariableop_resource_0"l
3while_lstmp_cell_2_matmul_1_readvariableop_resource5while_lstmp_cell_2_matmul_1_readvariableop_resource_0"l
3while_lstmp_cell_2_matmul_2_readvariableop_resource5while_lstmp_cell_2_matmul_2_readvariableop_resource_0"h
1while_lstmp_cell_2_matmul_readvariableop_resource3while_lstmp_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : 2V
)while/lstmp_cell_2/BiasAdd/ReadVariableOp)while/lstmp_cell_2/BiasAdd/ReadVariableOp2T
(while/lstmp_cell_2/MatMul/ReadVariableOp(while/lstmp_cell_2/MatMul/ReadVariableOp2X
*while/lstmp_cell_2/MatMul_1/ReadVariableOp*while/lstmp_cell_2/MatMul_1/ReadVariableOp2X
*while/lstmp_cell_2/MatMul_2/ReadVariableOp*while/lstmp_cell_2/MatMul_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
ã
í
while_cond_2876
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_12
.while_while_cond_2876___redundant_placeholder02
.while_while_cond_2876___redundant_placeholder12
.while_while_cond_2876___redundant_placeholder22
.while_while_cond_2876___redundant_placeholder32
.while_while_cond_2876___redundant_placeholder4
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
ª
Ð
&__inference_lstmp_2_layer_call_fn_2815

inputs
unknown:	
	unknown_0:	@
	unknown_1:	
	unknown_2:@@
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstmp_2_layer_call_and_return_conditional_losses_1925s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£

ø
%__inference_dlstmp_layer_call_fn_2156

inputs
unknown:	
	unknown_0:	@
	unknown_1:	
	unknown_2:@@
	unknown_3:	@
	unknown_4:	@
	unknown_5:	
	unknown_6:@@
	unknown_7:@
	unknown_8:
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dlstmp_layer_call_and_return_conditional_losses_1547o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ö
"__inference_signature_wrapper_2127
input_2
unknown:	
	unknown_0:	@
	unknown_1:	
	unknown_2:@@
	unknown_3:	@
	unknown_4:	@
	unknown_5:	
	unknown_6:@@
	unknown_7:@
	unknown_8:
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__wrapped_model_1201o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
¦

ù
%__inference_dlstmp_layer_call_fn_2038
input_2
unknown:	
	unknown_0:	@
	unknown_1:	
	unknown_2:@@
	unknown_3:	@
	unknown_4:	@
	unknown_5:	
	unknown_6:@@
	unknown_7:@
	unknown_8:
identity¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dlstmp_layer_call_and_return_conditional_losses_1990o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
ã
í
while_cond_1658
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_12
.while_while_cond_1658___redundant_placeholder02
.while_while_cond_1658___redundant_placeholder12
.while_while_cond_1658___redundant_placeholder22
.while_while_cond_1658___redundant_placeholder32
.while_while_cond_1658___redundant_placeholder4
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
æM

A__inference_lstmp_2_layer_call_and_return_conditional_losses_1358

inputs>
+lstmp_cell_2_matmul_readvariableop_resource:	@
-lstmp_cell_2_matmul_1_readvariableop_resource:	@;
,lstmp_cell_2_biasadd_readvariableop_resource:	?
-lstmp_cell_2_matmul_2_readvariableop_resource:@@
identity¢#lstmp_cell_2/BiasAdd/ReadVariableOp¢"lstmp_cell_2/MatMul/ReadVariableOp¢$lstmp_cell_2/MatMul_1/ReadVariableOp¢$lstmp_cell_2/MatMul_2/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstmp_cell_2/MatMul/ReadVariableOpReadVariableOp+lstmp_cell_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstmp_cell_2/MatMulMatMulstrided_slice_2:output:0*lstmp_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstmp_cell_2/MatMul_1/ReadVariableOpReadVariableOp-lstmp_cell_2_matmul_1_readvariableop_resource*
_output_shapes
:	@*
dtype0
lstmp_cell_2/MatMul_1MatMulzeros:output:0,lstmp_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstmp_cell_2/addAddV2lstmp_cell_2/MatMul:product:0lstmp_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstmp_cell_2/BiasAdd/ReadVariableOpReadVariableOp,lstmp_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstmp_cell_2/BiasAddBiasAddlstmp_cell_2/add:z:0+lstmp_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstmp_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstmp_cell_2/splitSplit%lstmp_cell_2/split/split_dim:output:0lstmp_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*
	num_splitn
lstmp_cell_2/SigmoidSigmoidlstmp_cell_2/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@p
lstmp_cell_2/Sigmoid_1Sigmoidlstmp_cell_2/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
lstmp_cell_2/mulMullstmp_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
lstmp_cell_2/TanhTanhlstmp_cell_2/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
lstmp_cell_2/mul_1Mullstmp_cell_2/Sigmoid:y:0lstmp_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
lstmp_cell_2/add_1AddV2lstmp_cell_2/mul:z:0lstmp_cell_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@p
lstmp_cell_2/Sigmoid_2Sigmoidlstmp_cell_2/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
lstmp_cell_2/Tanh_1Tanhlstmp_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_cell_2/mul_2Mullstmp_cell_2/Sigmoid_2:y:0lstmp_cell_2/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
$lstmp_cell_2/MatMul_2/ReadVariableOpReadVariableOp-lstmp_cell_2_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype0
lstmp_cell_2/MatMul_2MatMulzeros:output:0,lstmp_cell_2/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ³
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstmp_cell_2_matmul_readvariableop_resource-lstmp_cell_2_matmul_1_readvariableop_resource,lstmp_cell_2_biasadd_readvariableop_resource-lstmp_cell_2_matmul_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *&
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1270*
condR
while_cond_1269*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ç
NoOpNoOp$^lstmp_cell_2/BiasAdd/ReadVariableOp#^lstmp_cell_2/MatMul/ReadVariableOp%^lstmp_cell_2/MatMul_1/ReadVariableOp%^lstmp_cell_2/MatMul_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : 2J
#lstmp_cell_2/BiasAdd/ReadVariableOp#lstmp_cell_2/BiasAdd/ReadVariableOp2H
"lstmp_cell_2/MatMul/ReadVariableOp"lstmp_cell_2/MatMul/ReadVariableOp2L
$lstmp_cell_2/MatMul_1/ReadVariableOp$lstmp_cell_2/MatMul_1/ReadVariableOp2L
$lstmp_cell_2/MatMul_2/ReadVariableOp$lstmp_cell_2/MatMul_2/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦
·
dlstmp_lstmp_3_while_cond_1105:
6dlstmp_lstmp_3_while_dlstmp_lstmp_3_while_loop_counter@
<dlstmp_lstmp_3_while_dlstmp_lstmp_3_while_maximum_iterations$
 dlstmp_lstmp_3_while_placeholder&
"dlstmp_lstmp_3_while_placeholder_1&
"dlstmp_lstmp_3_while_placeholder_2&
"dlstmp_lstmp_3_while_placeholder_3<
8dlstmp_lstmp_3_while_less_dlstmp_lstmp_3_strided_slice_1P
Ldlstmp_lstmp_3_while_dlstmp_lstmp_3_while_cond_1105___redundant_placeholder0P
Ldlstmp_lstmp_3_while_dlstmp_lstmp_3_while_cond_1105___redundant_placeholder1P
Ldlstmp_lstmp_3_while_dlstmp_lstmp_3_while_cond_1105___redundant_placeholder2P
Ldlstmp_lstmp_3_while_dlstmp_lstmp_3_while_cond_1105___redundant_placeholder3P
Ldlstmp_lstmp_3_while_dlstmp_lstmp_3_while_cond_1105___redundant_placeholder4!
dlstmp_lstmp_3_while_identity

dlstmp/lstmp_3/while/LessLess dlstmp_lstmp_3_while_placeholder8dlstmp_lstmp_3_while_less_dlstmp_lstmp_3_strided_slice_1*
T0*
_output_shapes
: i
dlstmp/lstmp_3/while/IdentityIdentitydlstmp/lstmp_3/while/Less:z:0*
T0
*
_output_shapes
: "G
dlstmp_lstmp_3_while_identity&dlstmp/lstmp_3/while/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
¾I
£
lstmp_3_while_body_2694,
(lstmp_3_while_lstmp_3_while_loop_counter2
.lstmp_3_while_lstmp_3_while_maximum_iterations
lstmp_3_while_placeholder
lstmp_3_while_placeholder_1
lstmp_3_while_placeholder_2
lstmp_3_while_placeholder_3+
'lstmp_3_while_lstmp_3_strided_slice_1_0g
clstmp_3_while_tensorarrayv2read_tensorlistgetitem_lstmp_3_tensorarrayunstack_tensorlistfromtensor_0N
;lstmp_3_while_lstmp_cell_3_matmul_readvariableop_resource_0:	@P
=lstmp_3_while_lstmp_cell_3_matmul_1_readvariableop_resource_0:	@K
<lstmp_3_while_lstmp_cell_3_biasadd_readvariableop_resource_0:	O
=lstmp_3_while_lstmp_cell_3_matmul_2_readvariableop_resource_0:@@
lstmp_3_while_identity
lstmp_3_while_identity_1
lstmp_3_while_identity_2
lstmp_3_while_identity_3
lstmp_3_while_identity_4
lstmp_3_while_identity_5)
%lstmp_3_while_lstmp_3_strided_slice_1e
alstmp_3_while_tensorarrayv2read_tensorlistgetitem_lstmp_3_tensorarrayunstack_tensorlistfromtensorL
9lstmp_3_while_lstmp_cell_3_matmul_readvariableop_resource:	@N
;lstmp_3_while_lstmp_cell_3_matmul_1_readvariableop_resource:	@I
:lstmp_3_while_lstmp_cell_3_biasadd_readvariableop_resource:	M
;lstmp_3_while_lstmp_cell_3_matmul_2_readvariableop_resource:@@¢1lstmp_3/while/lstmp_cell_3/BiasAdd/ReadVariableOp¢0lstmp_3/while/lstmp_cell_3/MatMul/ReadVariableOp¢2lstmp_3/while/lstmp_cell_3/MatMul_1/ReadVariableOp¢2lstmp_3/while/lstmp_cell_3/MatMul_2/ReadVariableOp
?lstmp_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Î
1lstmp_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstmp_3_while_tensorarrayv2read_tensorlistgetitem_lstmp_3_tensorarrayunstack_tensorlistfromtensor_0lstmp_3_while_placeholderHlstmp_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0­
0lstmp_3/while/lstmp_cell_3/MatMul/ReadVariableOpReadVariableOp;lstmp_3_while_lstmp_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	@*
dtype0Ò
!lstmp_3/while/lstmp_cell_3/MatMulMatMul8lstmp_3/while/TensorArrayV2Read/TensorListGetItem:item:08lstmp_3/while/lstmp_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
2lstmp_3/while/lstmp_cell_3/MatMul_1/ReadVariableOpReadVariableOp=lstmp_3_while_lstmp_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes
:	@*
dtype0¹
#lstmp_3/while/lstmp_cell_3/MatMul_1MatMullstmp_3_while_placeholder_2:lstmp_3/while/lstmp_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
lstmp_3/while/lstmp_cell_3/addAddV2+lstmp_3/while/lstmp_cell_3/MatMul:product:0-lstmp_3/while/lstmp_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
1lstmp_3/while/lstmp_cell_3/BiasAdd/ReadVariableOpReadVariableOp<lstmp_3_while_lstmp_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0¿
"lstmp_3/while/lstmp_cell_3/BiasAddBiasAdd"lstmp_3/while/lstmp_cell_3/add:z:09lstmp_3/while/lstmp_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
*lstmp_3/while/lstmp_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstmp_3/while/lstmp_cell_3/splitSplit3lstmp_3/while/lstmp_cell_3/split/split_dim:output:0+lstmp_3/while/lstmp_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*
	num_split
"lstmp_3/while/lstmp_cell_3/SigmoidSigmoid)lstmp_3/while/lstmp_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
$lstmp_3/while/lstmp_cell_3/Sigmoid_1Sigmoid)lstmp_3/while/lstmp_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_3/while/lstmp_cell_3/mulMul(lstmp_3/while/lstmp_cell_3/Sigmoid_1:y:0lstmp_3_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_3/while/lstmp_cell_3/TanhTanh)lstmp_3/while/lstmp_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
 lstmp_3/while/lstmp_cell_3/mul_1Mul&lstmp_3/while/lstmp_cell_3/Sigmoid:y:0#lstmp_3/while/lstmp_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
 lstmp_3/while/lstmp_cell_3/add_1AddV2"lstmp_3/while/lstmp_cell_3/mul:z:0$lstmp_3/while/lstmp_cell_3/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
$lstmp_3/while/lstmp_cell_3/Sigmoid_2Sigmoid)lstmp_3/while/lstmp_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
!lstmp_3/while/lstmp_cell_3/Tanh_1Tanh$lstmp_3/while/lstmp_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ª
 lstmp_3/while/lstmp_cell_3/mul_2Mul(lstmp_3/while/lstmp_cell_3/Sigmoid_2:y:0%lstmp_3/while/lstmp_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@°
2lstmp_3/while/lstmp_cell_3/MatMul_2/ReadVariableOpReadVariableOp=lstmp_3_while_lstmp_cell_3_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¸
#lstmp_3/while/lstmp_cell_3/MatMul_2MatMullstmp_3_while_placeholder_2:lstmp_3/while/lstmp_cell_3/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
8lstmp_3/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
2lstmp_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstmp_3_while_placeholder_1Alstmp_3/while/TensorArrayV2Write/TensorListSetItem/index:output:0$lstmp_3/while/lstmp_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒU
lstmp_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstmp_3/while/addAddV2lstmp_3_while_placeholderlstmp_3/while/add/y:output:0*
T0*
_output_shapes
: W
lstmp_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstmp_3/while/add_1AddV2(lstmp_3_while_lstmp_3_while_loop_counterlstmp_3/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstmp_3/while/IdentityIdentitylstmp_3/while/add_1:z:0^lstmp_3/while/NoOp*
T0*
_output_shapes
: 
lstmp_3/while/Identity_1Identity.lstmp_3_while_lstmp_3_while_maximum_iterations^lstmp_3/while/NoOp*
T0*
_output_shapes
: q
lstmp_3/while/Identity_2Identitylstmp_3/while/add:z:0^lstmp_3/while/NoOp*
T0*
_output_shapes
: 
lstmp_3/while/Identity_3IdentityBlstmp_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstmp_3/while/NoOp*
T0*
_output_shapes
: 
lstmp_3/while/Identity_4Identity-lstmp_3/while/lstmp_cell_3/MatMul_2:product:0^lstmp_3/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_3/while/Identity_5Identity$lstmp_3/while/lstmp_cell_3/add_1:z:0^lstmp_3/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
lstmp_3/while/NoOpNoOp2^lstmp_3/while/lstmp_cell_3/BiasAdd/ReadVariableOp1^lstmp_3/while/lstmp_cell_3/MatMul/ReadVariableOp3^lstmp_3/while/lstmp_cell_3/MatMul_1/ReadVariableOp3^lstmp_3/while/lstmp_cell_3/MatMul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstmp_3_while_identitylstmp_3/while/Identity:output:0"=
lstmp_3_while_identity_1!lstmp_3/while/Identity_1:output:0"=
lstmp_3_while_identity_2!lstmp_3/while/Identity_2:output:0"=
lstmp_3_while_identity_3!lstmp_3/while/Identity_3:output:0"=
lstmp_3_while_identity_4!lstmp_3/while/Identity_4:output:0"=
lstmp_3_while_identity_5!lstmp_3/while/Identity_5:output:0"P
%lstmp_3_while_lstmp_3_strided_slice_1'lstmp_3_while_lstmp_3_strided_slice_1_0"z
:lstmp_3_while_lstmp_cell_3_biasadd_readvariableop_resource<lstmp_3_while_lstmp_cell_3_biasadd_readvariableop_resource_0"|
;lstmp_3_while_lstmp_cell_3_matmul_1_readvariableop_resource=lstmp_3_while_lstmp_cell_3_matmul_1_readvariableop_resource_0"|
;lstmp_3_while_lstmp_cell_3_matmul_2_readvariableop_resource=lstmp_3_while_lstmp_cell_3_matmul_2_readvariableop_resource_0"x
9lstmp_3_while_lstmp_cell_3_matmul_readvariableop_resource;lstmp_3_while_lstmp_cell_3_matmul_readvariableop_resource_0"È
alstmp_3_while_tensorarrayv2read_tensorlistgetitem_lstmp_3_tensorarrayunstack_tensorlistfromtensorclstmp_3_while_tensorarrayv2read_tensorlistgetitem_lstmp_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : 2f
1lstmp_3/while/lstmp_cell_3/BiasAdd/ReadVariableOp1lstmp_3/while/lstmp_cell_3/BiasAdd/ReadVariableOp2d
0lstmp_3/while/lstmp_cell_3/MatMul/ReadVariableOp0lstmp_3/while/lstmp_cell_3/MatMul/ReadVariableOp2h
2lstmp_3/while/lstmp_cell_3/MatMul_1/ReadVariableOp2lstmp_3/while/lstmp_cell_3/MatMul_1/ReadVariableOp2h
2lstmp_3/while/lstmp_cell_3/MatMul_2/ReadVariableOp2lstmp_3/while/lstmp_cell_3/MatMul_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
ìN

A__inference_lstmp_3_layer_call_and_return_conditional_losses_1520

inputs>
+lstmp_cell_3_matmul_readvariableop_resource:	@@
-lstmp_cell_3_matmul_1_readvariableop_resource:	@;
,lstmp_cell_3_biasadd_readvariableop_resource:	?
-lstmp_cell_3_matmul_2_readvariableop_resource:@@
identity¢#lstmp_cell_3/BiasAdd/ReadVariableOp¢"lstmp_cell_3/MatMul/ReadVariableOp¢$lstmp_cell_3/MatMul_1/ReadVariableOp¢$lstmp_cell_3/MatMul_2/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
"lstmp_cell_3/MatMul/ReadVariableOpReadVariableOp+lstmp_cell_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
lstmp_cell_3/MatMulMatMulstrided_slice_2:output:0*lstmp_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstmp_cell_3/MatMul_1/ReadVariableOpReadVariableOp-lstmp_cell_3_matmul_1_readvariableop_resource*
_output_shapes
:	@*
dtype0
lstmp_cell_3/MatMul_1MatMulzeros:output:0,lstmp_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstmp_cell_3/addAddV2lstmp_cell_3/MatMul:product:0lstmp_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstmp_cell_3/BiasAdd/ReadVariableOpReadVariableOp,lstmp_cell_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstmp_cell_3/BiasAddBiasAddlstmp_cell_3/add:z:0+lstmp_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstmp_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstmp_cell_3/splitSplit%lstmp_cell_3/split/split_dim:output:0lstmp_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*
	num_splitn
lstmp_cell_3/SigmoidSigmoidlstmp_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@p
lstmp_cell_3/Sigmoid_1Sigmoidlstmp_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
lstmp_cell_3/mulMullstmp_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
lstmp_cell_3/TanhTanhlstmp_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
lstmp_cell_3/mul_1Mullstmp_cell_3/Sigmoid:y:0lstmp_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
lstmp_cell_3/add_1AddV2lstmp_cell_3/mul:z:0lstmp_cell_3/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@p
lstmp_cell_3/Sigmoid_2Sigmoidlstmp_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
lstmp_cell_3/Tanh_1Tanhlstmp_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_cell_3/mul_2Mullstmp_cell_3/Sigmoid_2:y:0lstmp_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
$lstmp_cell_3/MatMul_2/ReadVariableOpReadVariableOp-lstmp_cell_3_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype0
lstmp_cell_3/MatMul_2MatMulzeros:output:0,lstmp_cell_3/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ³
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstmp_cell_3_matmul_readvariableop_resource-lstmp_cell_3_matmul_1_readvariableop_resource,lstmp_cell_3_biasadd_readvariableop_resource-lstmp_cell_3_matmul_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *&
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1431*
condR
while_cond_1430*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ç
NoOpNoOp$^lstmp_cell_3/BiasAdd/ReadVariableOp#^lstmp_cell_3/MatMul/ReadVariableOp%^lstmp_cell_3/MatMul_1/ReadVariableOp%^lstmp_cell_3/MatMul_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : : : 2J
#lstmp_cell_3/BiasAdd/ReadVariableOp#lstmp_cell_3/BiasAdd/ReadVariableOp2H
"lstmp_cell_3/MatMul/ReadVariableOp"lstmp_cell_3/MatMul/ReadVariableOp2L
$lstmp_cell_3/MatMul_1/ReadVariableOp$lstmp_cell_3/MatMul_1/ReadVariableOp2L
$lstmp_cell_3/MatMul_2/ReadVariableOp$lstmp_cell_3/MatMul_2/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
?


while_body_3358
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstmp_cell_3_matmul_readvariableop_resource_0:	@H
5while_lstmp_cell_3_matmul_1_readvariableop_resource_0:	@C
4while_lstmp_cell_3_biasadd_readvariableop_resource_0:	G
5while_lstmp_cell_3_matmul_2_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstmp_cell_3_matmul_readvariableop_resource:	@F
3while_lstmp_cell_3_matmul_1_readvariableop_resource:	@A
2while_lstmp_cell_3_biasadd_readvariableop_resource:	E
3while_lstmp_cell_3_matmul_2_readvariableop_resource:@@¢)while/lstmp_cell_3/BiasAdd/ReadVariableOp¢(while/lstmp_cell_3/MatMul/ReadVariableOp¢*while/lstmp_cell_3/MatMul_1/ReadVariableOp¢*while/lstmp_cell_3/MatMul_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0
(while/lstmp_cell_3/MatMul/ReadVariableOpReadVariableOp3while_lstmp_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	@*
dtype0º
while/lstmp_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstmp_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
*while/lstmp_cell_3/MatMul_1/ReadVariableOpReadVariableOp5while_lstmp_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes
:	@*
dtype0¡
while/lstmp_cell_3/MatMul_1MatMulwhile_placeholder_22while/lstmp_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstmp_cell_3/addAddV2#while/lstmp_cell_3/MatMul:product:0%while/lstmp_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstmp_cell_3/BiasAdd/ReadVariableOpReadVariableOp4while_lstmp_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstmp_cell_3/BiasAddBiasAddwhile/lstmp_cell_3/add:z:01while/lstmp_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstmp_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstmp_cell_3/splitSplit+while/lstmp_cell_3/split/split_dim:output:0#while/lstmp_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*
	num_splitz
while/lstmp_cell_3/SigmoidSigmoid!while/lstmp_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
while/lstmp_cell_3/Sigmoid_1Sigmoid!while/lstmp_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_3/mulMul while/lstmp_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@t
while/lstmp_cell_3/TanhTanh!while/lstmp_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_3/mul_1Mulwhile/lstmp_cell_3/Sigmoid:y:0while/lstmp_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_3/add_1AddV2while/lstmp_cell_3/mul:z:0while/lstmp_cell_3/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
while/lstmp_cell_3/Sigmoid_2Sigmoid!while/lstmp_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
while/lstmp_cell_3/Tanh_1Tanhwhile/lstmp_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_3/mul_2Mul while/lstmp_cell_3/Sigmoid_2:y:0while/lstmp_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
*while/lstmp_cell_3/MatMul_2/ReadVariableOpReadVariableOp5while_lstmp_cell_3_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype0 
while/lstmp_cell_3/MatMul_2MatMulwhile_placeholder_22while/lstmp_cell_3/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : í
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstmp_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity%while/lstmp_cell_3/MatMul_2:product:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/Identity_5Identitywhile/lstmp_cell_3/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ý

while/NoOpNoOp*^while/lstmp_cell_3/BiasAdd/ReadVariableOp)^while/lstmp_cell_3/MatMul/ReadVariableOp+^while/lstmp_cell_3/MatMul_1/ReadVariableOp+^while/lstmp_cell_3/MatMul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstmp_cell_3_biasadd_readvariableop_resource4while_lstmp_cell_3_biasadd_readvariableop_resource_0"l
3while_lstmp_cell_3_matmul_1_readvariableop_resource5while_lstmp_cell_3_matmul_1_readvariableop_resource_0"l
3while_lstmp_cell_3_matmul_2_readvariableop_resource5while_lstmp_cell_3_matmul_2_readvariableop_resource_0"h
1while_lstmp_cell_3_matmul_readvariableop_resource3while_lstmp_cell_3_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : 2V
)while/lstmp_cell_3/BiasAdd/ReadVariableOp)while/lstmp_cell_3/BiasAdd/ReadVariableOp2T
(while/lstmp_cell_3/MatMul/ReadVariableOp(while/lstmp_cell_3/MatMul/ReadVariableOp2X
*while/lstmp_cell_3/MatMul_1/ReadVariableOp*while/lstmp_cell_3/MatMul_1/ReadVariableOp2X
*while/lstmp_cell_3/MatMul_2/ReadVariableOp*while/lstmp_cell_3/MatMul_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
¢
Ð
&__inference_lstmp_3_layer_call_fn_3130

inputs
unknown:	@
	unknown_0:	@
	unknown_1:	
	unknown_2:@@
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstmp_3_layer_call_and_return_conditional_losses_1520o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
H
£
lstmp_2_while_body_2243,
(lstmp_2_while_lstmp_2_while_loop_counter2
.lstmp_2_while_lstmp_2_while_maximum_iterations
lstmp_2_while_placeholder
lstmp_2_while_placeholder_1
lstmp_2_while_placeholder_2
lstmp_2_while_placeholder_3+
'lstmp_2_while_lstmp_2_strided_slice_1_0g
clstmp_2_while_tensorarrayv2read_tensorlistgetitem_lstmp_2_tensorarrayunstack_tensorlistfromtensor_0N
;lstmp_2_while_lstmp_cell_2_matmul_readvariableop_resource_0:	P
=lstmp_2_while_lstmp_cell_2_matmul_1_readvariableop_resource_0:	@K
<lstmp_2_while_lstmp_cell_2_biasadd_readvariableop_resource_0:	O
=lstmp_2_while_lstmp_cell_2_matmul_2_readvariableop_resource_0:@@
lstmp_2_while_identity
lstmp_2_while_identity_1
lstmp_2_while_identity_2
lstmp_2_while_identity_3
lstmp_2_while_identity_4
lstmp_2_while_identity_5)
%lstmp_2_while_lstmp_2_strided_slice_1e
alstmp_2_while_tensorarrayv2read_tensorlistgetitem_lstmp_2_tensorarrayunstack_tensorlistfromtensorL
9lstmp_2_while_lstmp_cell_2_matmul_readvariableop_resource:	N
;lstmp_2_while_lstmp_cell_2_matmul_1_readvariableop_resource:	@I
:lstmp_2_while_lstmp_cell_2_biasadd_readvariableop_resource:	M
;lstmp_2_while_lstmp_cell_2_matmul_2_readvariableop_resource:@@¢1lstmp_2/while/lstmp_cell_2/BiasAdd/ReadVariableOp¢0lstmp_2/while/lstmp_cell_2/MatMul/ReadVariableOp¢2lstmp_2/while/lstmp_cell_2/MatMul_1/ReadVariableOp¢2lstmp_2/while/lstmp_cell_2/MatMul_2/ReadVariableOp
?lstmp_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Î
1lstmp_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstmp_2_while_tensorarrayv2read_tensorlistgetitem_lstmp_2_tensorarrayunstack_tensorlistfromtensor_0lstmp_2_while_placeholderHlstmp_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0­
0lstmp_2/while/lstmp_cell_2/MatMul/ReadVariableOpReadVariableOp;lstmp_2_while_lstmp_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ò
!lstmp_2/while/lstmp_cell_2/MatMulMatMul8lstmp_2/while/TensorArrayV2Read/TensorListGetItem:item:08lstmp_2/while/lstmp_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
2lstmp_2/while/lstmp_cell_2/MatMul_1/ReadVariableOpReadVariableOp=lstmp_2_while_lstmp_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes
:	@*
dtype0¹
#lstmp_2/while/lstmp_cell_2/MatMul_1MatMullstmp_2_while_placeholder_2:lstmp_2/while/lstmp_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
lstmp_2/while/lstmp_cell_2/addAddV2+lstmp_2/while/lstmp_cell_2/MatMul:product:0-lstmp_2/while/lstmp_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
1lstmp_2/while/lstmp_cell_2/BiasAdd/ReadVariableOpReadVariableOp<lstmp_2_while_lstmp_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0¿
"lstmp_2/while/lstmp_cell_2/BiasAddBiasAdd"lstmp_2/while/lstmp_cell_2/add:z:09lstmp_2/while/lstmp_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
*lstmp_2/while/lstmp_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstmp_2/while/lstmp_cell_2/splitSplit3lstmp_2/while/lstmp_cell_2/split/split_dim:output:0+lstmp_2/while/lstmp_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*
	num_split
"lstmp_2/while/lstmp_cell_2/SigmoidSigmoid)lstmp_2/while/lstmp_cell_2/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
$lstmp_2/while/lstmp_cell_2/Sigmoid_1Sigmoid)lstmp_2/while/lstmp_cell_2/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_2/while/lstmp_cell_2/mulMul(lstmp_2/while/lstmp_cell_2/Sigmoid_1:y:0lstmp_2_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_2/while/lstmp_cell_2/TanhTanh)lstmp_2/while/lstmp_cell_2/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
 lstmp_2/while/lstmp_cell_2/mul_1Mul&lstmp_2/while/lstmp_cell_2/Sigmoid:y:0#lstmp_2/while/lstmp_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
 lstmp_2/while/lstmp_cell_2/add_1AddV2"lstmp_2/while/lstmp_cell_2/mul:z:0$lstmp_2/while/lstmp_cell_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
$lstmp_2/while/lstmp_cell_2/Sigmoid_2Sigmoid)lstmp_2/while/lstmp_cell_2/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
!lstmp_2/while/lstmp_cell_2/Tanh_1Tanh$lstmp_2/while/lstmp_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ª
 lstmp_2/while/lstmp_cell_2/mul_2Mul(lstmp_2/while/lstmp_cell_2/Sigmoid_2:y:0%lstmp_2/while/lstmp_cell_2/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@°
2lstmp_2/while/lstmp_cell_2/MatMul_2/ReadVariableOpReadVariableOp=lstmp_2_while_lstmp_cell_2_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¸
#lstmp_2/while/lstmp_cell_2/MatMul_2MatMullstmp_2_while_placeholder_2:lstmp_2/while/lstmp_cell_2/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@å
2lstmp_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstmp_2_while_placeholder_1lstmp_2_while_placeholder$lstmp_2/while/lstmp_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒU
lstmp_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstmp_2/while/addAddV2lstmp_2_while_placeholderlstmp_2/while/add/y:output:0*
T0*
_output_shapes
: W
lstmp_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstmp_2/while/add_1AddV2(lstmp_2_while_lstmp_2_while_loop_counterlstmp_2/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstmp_2/while/IdentityIdentitylstmp_2/while/add_1:z:0^lstmp_2/while/NoOp*
T0*
_output_shapes
: 
lstmp_2/while/Identity_1Identity.lstmp_2_while_lstmp_2_while_maximum_iterations^lstmp_2/while/NoOp*
T0*
_output_shapes
: q
lstmp_2/while/Identity_2Identitylstmp_2/while/add:z:0^lstmp_2/while/NoOp*
T0*
_output_shapes
: 
lstmp_2/while/Identity_3IdentityBlstmp_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstmp_2/while/NoOp*
T0*
_output_shapes
: 
lstmp_2/while/Identity_4Identity-lstmp_2/while/lstmp_cell_2/MatMul_2:product:0^lstmp_2/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_2/while/Identity_5Identity$lstmp_2/while/lstmp_cell_2/add_1:z:0^lstmp_2/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
lstmp_2/while/NoOpNoOp2^lstmp_2/while/lstmp_cell_2/BiasAdd/ReadVariableOp1^lstmp_2/while/lstmp_cell_2/MatMul/ReadVariableOp3^lstmp_2/while/lstmp_cell_2/MatMul_1/ReadVariableOp3^lstmp_2/while/lstmp_cell_2/MatMul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstmp_2_while_identitylstmp_2/while/Identity:output:0"=
lstmp_2_while_identity_1!lstmp_2/while/Identity_1:output:0"=
lstmp_2_while_identity_2!lstmp_2/while/Identity_2:output:0"=
lstmp_2_while_identity_3!lstmp_2/while/Identity_3:output:0"=
lstmp_2_while_identity_4!lstmp_2/while/Identity_4:output:0"=
lstmp_2_while_identity_5!lstmp_2/while/Identity_5:output:0"P
%lstmp_2_while_lstmp_2_strided_slice_1'lstmp_2_while_lstmp_2_strided_slice_1_0"z
:lstmp_2_while_lstmp_cell_2_biasadd_readvariableop_resource<lstmp_2_while_lstmp_cell_2_biasadd_readvariableop_resource_0"|
;lstmp_2_while_lstmp_cell_2_matmul_1_readvariableop_resource=lstmp_2_while_lstmp_cell_2_matmul_1_readvariableop_resource_0"|
;lstmp_2_while_lstmp_cell_2_matmul_2_readvariableop_resource=lstmp_2_while_lstmp_cell_2_matmul_2_readvariableop_resource_0"x
9lstmp_2_while_lstmp_cell_2_matmul_readvariableop_resource;lstmp_2_while_lstmp_cell_2_matmul_readvariableop_resource_0"È
alstmp_2_while_tensorarrayv2read_tensorlistgetitem_lstmp_2_tensorarrayunstack_tensorlistfromtensorclstmp_2_while_tensorarrayv2read_tensorlistgetitem_lstmp_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : 2f
1lstmp_2/while/lstmp_cell_2/BiasAdd/ReadVariableOp1lstmp_2/while/lstmp_cell_2/BiasAdd/ReadVariableOp2d
0lstmp_2/while/lstmp_cell_2/MatMul/ReadVariableOp0lstmp_2/while/lstmp_cell_2/MatMul/ReadVariableOp2h
2lstmp_2/while/lstmp_cell_2/MatMul_1/ReadVariableOp2lstmp_2/while/lstmp_cell_2/MatMul_1/ReadVariableOp2h
2lstmp_2/while/lstmp_cell_2/MatMul_2/ReadVariableOp2lstmp_2/while/lstmp_cell_2/MatMul_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
 
ÿ
@__inference_dlstmp_layer_call_and_return_conditional_losses_1990

inputs
lstmp_2_1966:	
lstmp_2_1968:	@
lstmp_2_1970:	
lstmp_2_1972:@@
lstmp_3_1975:	@
lstmp_3_1977:	@
lstmp_3_1979:	
lstmp_3_1981:@@
dense_1_1984:@
dense_1_1986:
identity¢dense_1/StatefulPartitionedCall¢lstmp_2/StatefulPartitionedCall¢lstmp_3/StatefulPartitionedCall
lstmp_2/StatefulPartitionedCallStatefulPartitionedCallinputslstmp_2_1966lstmp_2_1968lstmp_2_1970lstmp_2_1972*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstmp_2_layer_call_and_return_conditional_losses_1925«
lstmp_3/StatefulPartitionedCallStatefulPartitionedCall(lstmp_2/StatefulPartitionedCall:output:0lstmp_3_1975lstmp_3_1977lstmp_3_1979lstmp_3_1981*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstmp_3_layer_call_and_return_conditional_losses_1748
dense_1/StatefulPartitionedCallStatefulPartitionedCall(lstmp_3/StatefulPartitionedCall:output:0dense_1_1984dense_1_1986*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_1540w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
NoOpNoOp ^dense_1/StatefulPartitionedCall ^lstmp_2/StatefulPartitionedCall ^lstmp_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
lstmp_2/StatefulPartitionedCalllstmp_2/StatefulPartitionedCall2B
lstmp_3/StatefulPartitionedCalllstmp_3/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿

&__inference_dense_1_layer_call_fn_3458

inputs
unknown:@
	unknown_0:
identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_1540o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ìN

A__inference_lstmp_3_layer_call_and_return_conditional_losses_3447

inputs>
+lstmp_cell_3_matmul_readvariableop_resource:	@@
-lstmp_cell_3_matmul_1_readvariableop_resource:	@;
,lstmp_cell_3_biasadd_readvariableop_resource:	?
-lstmp_cell_3_matmul_2_readvariableop_resource:@@
identity¢#lstmp_cell_3/BiasAdd/ReadVariableOp¢"lstmp_cell_3/MatMul/ReadVariableOp¢$lstmp_cell_3/MatMul_1/ReadVariableOp¢$lstmp_cell_3/MatMul_2/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
"lstmp_cell_3/MatMul/ReadVariableOpReadVariableOp+lstmp_cell_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
lstmp_cell_3/MatMulMatMulstrided_slice_2:output:0*lstmp_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstmp_cell_3/MatMul_1/ReadVariableOpReadVariableOp-lstmp_cell_3_matmul_1_readvariableop_resource*
_output_shapes
:	@*
dtype0
lstmp_cell_3/MatMul_1MatMulzeros:output:0,lstmp_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstmp_cell_3/addAddV2lstmp_cell_3/MatMul:product:0lstmp_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstmp_cell_3/BiasAdd/ReadVariableOpReadVariableOp,lstmp_cell_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstmp_cell_3/BiasAddBiasAddlstmp_cell_3/add:z:0+lstmp_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstmp_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstmp_cell_3/splitSplit%lstmp_cell_3/split/split_dim:output:0lstmp_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*
	num_splitn
lstmp_cell_3/SigmoidSigmoidlstmp_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@p
lstmp_cell_3/Sigmoid_1Sigmoidlstmp_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
lstmp_cell_3/mulMullstmp_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
lstmp_cell_3/TanhTanhlstmp_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
lstmp_cell_3/mul_1Mullstmp_cell_3/Sigmoid:y:0lstmp_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
lstmp_cell_3/add_1AddV2lstmp_cell_3/mul:z:0lstmp_cell_3/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@p
lstmp_cell_3/Sigmoid_2Sigmoidlstmp_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
lstmp_cell_3/Tanh_1Tanhlstmp_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_cell_3/mul_2Mullstmp_cell_3/Sigmoid_2:y:0lstmp_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
$lstmp_cell_3/MatMul_2/ReadVariableOpReadVariableOp-lstmp_cell_3_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype0
lstmp_cell_3/MatMul_2MatMulzeros:output:0,lstmp_cell_3/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ³
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstmp_cell_3_matmul_readvariableop_resource-lstmp_cell_3_matmul_1_readvariableop_resource,lstmp_cell_3_biasadd_readvariableop_resource-lstmp_cell_3_matmul_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *&
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3358*
condR
while_cond_3357*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ç
NoOpNoOp$^lstmp_cell_3/BiasAdd/ReadVariableOp#^lstmp_cell_3/MatMul/ReadVariableOp%^lstmp_cell_3/MatMul_1/ReadVariableOp%^lstmp_cell_3/MatMul_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : : : 2J
#lstmp_cell_3/BiasAdd/ReadVariableOp#lstmp_cell_3/BiasAdd/ReadVariableOp2H
"lstmp_cell_3/MatMul/ReadVariableOp"lstmp_cell_3/MatMul/ReadVariableOp2L
$lstmp_cell_3/MatMul_1/ReadVariableOp$lstmp_cell_3/MatMul_1/ReadVariableOp2L
$lstmp_cell_3/MatMul_2/ReadVariableOp$lstmp_cell_3/MatMul_2/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¢
Ð
&__inference_lstmp_3_layer_call_fn_3143

inputs
unknown:	@
	unknown_0:	@
	unknown_1:	
	unknown_2:@@
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstmp_3_layer_call_and_return_conditional_losses_1748o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
æM

A__inference_lstmp_2_layer_call_and_return_conditional_losses_1925

inputs>
+lstmp_cell_2_matmul_readvariableop_resource:	@
-lstmp_cell_2_matmul_1_readvariableop_resource:	@;
,lstmp_cell_2_biasadd_readvariableop_resource:	?
-lstmp_cell_2_matmul_2_readvariableop_resource:@@
identity¢#lstmp_cell_2/BiasAdd/ReadVariableOp¢"lstmp_cell_2/MatMul/ReadVariableOp¢$lstmp_cell_2/MatMul_1/ReadVariableOp¢$lstmp_cell_2/MatMul_2/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstmp_cell_2/MatMul/ReadVariableOpReadVariableOp+lstmp_cell_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstmp_cell_2/MatMulMatMulstrided_slice_2:output:0*lstmp_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstmp_cell_2/MatMul_1/ReadVariableOpReadVariableOp-lstmp_cell_2_matmul_1_readvariableop_resource*
_output_shapes
:	@*
dtype0
lstmp_cell_2/MatMul_1MatMulzeros:output:0,lstmp_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstmp_cell_2/addAddV2lstmp_cell_2/MatMul:product:0lstmp_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstmp_cell_2/BiasAdd/ReadVariableOpReadVariableOp,lstmp_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstmp_cell_2/BiasAddBiasAddlstmp_cell_2/add:z:0+lstmp_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstmp_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstmp_cell_2/splitSplit%lstmp_cell_2/split/split_dim:output:0lstmp_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*
	num_splitn
lstmp_cell_2/SigmoidSigmoidlstmp_cell_2/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@p
lstmp_cell_2/Sigmoid_1Sigmoidlstmp_cell_2/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
lstmp_cell_2/mulMullstmp_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
lstmp_cell_2/TanhTanhlstmp_cell_2/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
lstmp_cell_2/mul_1Mullstmp_cell_2/Sigmoid:y:0lstmp_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
lstmp_cell_2/add_1AddV2lstmp_cell_2/mul:z:0lstmp_cell_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@p
lstmp_cell_2/Sigmoid_2Sigmoidlstmp_cell_2/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
lstmp_cell_2/Tanh_1Tanhlstmp_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_cell_2/mul_2Mullstmp_cell_2/Sigmoid_2:y:0lstmp_cell_2/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
$lstmp_cell_2/MatMul_2/ReadVariableOpReadVariableOp-lstmp_cell_2_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype0
lstmp_cell_2/MatMul_2MatMulzeros:output:0,lstmp_cell_2/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ³
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstmp_cell_2_matmul_readvariableop_resource-lstmp_cell_2_matmul_1_readvariableop_resource,lstmp_cell_2_biasadd_readvariableop_resource-lstmp_cell_2_matmul_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *&
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1837*
condR
while_cond_1836*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ç
NoOpNoOp$^lstmp_cell_2/BiasAdd/ReadVariableOp#^lstmp_cell_2/MatMul/ReadVariableOp%^lstmp_cell_2/MatMul_1/ReadVariableOp%^lstmp_cell_2/MatMul_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : 2J
#lstmp_cell_2/BiasAdd/ReadVariableOp#lstmp_cell_2/BiasAdd/ReadVariableOp2H
"lstmp_cell_2/MatMul/ReadVariableOp"lstmp_cell_2/MatMul/ReadVariableOp2L
$lstmp_cell_2/MatMul_1/ReadVariableOp$lstmp_cell_2/MatMul_1/ReadVariableOp2L
$lstmp_cell_2/MatMul_2/ReadVariableOp$lstmp_cell_2/MatMul_2/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú=


while_body_2877
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstmp_cell_2_matmul_readvariableop_resource_0:	H
5while_lstmp_cell_2_matmul_1_readvariableop_resource_0:	@C
4while_lstmp_cell_2_biasadd_readvariableop_resource_0:	G
5while_lstmp_cell_2_matmul_2_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstmp_cell_2_matmul_readvariableop_resource:	F
3while_lstmp_cell_2_matmul_1_readvariableop_resource:	@A
2while_lstmp_cell_2_biasadd_readvariableop_resource:	E
3while_lstmp_cell_2_matmul_2_readvariableop_resource:@@¢)while/lstmp_cell_2/BiasAdd/ReadVariableOp¢(while/lstmp_cell_2/MatMul/ReadVariableOp¢*while/lstmp_cell_2/MatMul_1/ReadVariableOp¢*while/lstmp_cell_2/MatMul_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstmp_cell_2/MatMul/ReadVariableOpReadVariableOp3while_lstmp_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstmp_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstmp_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
*while/lstmp_cell_2/MatMul_1/ReadVariableOpReadVariableOp5while_lstmp_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes
:	@*
dtype0¡
while/lstmp_cell_2/MatMul_1MatMulwhile_placeholder_22while/lstmp_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstmp_cell_2/addAddV2#while/lstmp_cell_2/MatMul:product:0%while/lstmp_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstmp_cell_2/BiasAdd/ReadVariableOpReadVariableOp4while_lstmp_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstmp_cell_2/BiasAddBiasAddwhile/lstmp_cell_2/add:z:01while/lstmp_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstmp_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstmp_cell_2/splitSplit+while/lstmp_cell_2/split/split_dim:output:0#while/lstmp_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*
	num_splitz
while/lstmp_cell_2/SigmoidSigmoid!while/lstmp_cell_2/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
while/lstmp_cell_2/Sigmoid_1Sigmoid!while/lstmp_cell_2/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_2/mulMul while/lstmp_cell_2/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@t
while/lstmp_cell_2/TanhTanh!while/lstmp_cell_2/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_2/mul_1Mulwhile/lstmp_cell_2/Sigmoid:y:0while/lstmp_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_2/add_1AddV2while/lstmp_cell_2/mul:z:0while/lstmp_cell_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
while/lstmp_cell_2/Sigmoid_2Sigmoid!while/lstmp_cell_2/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
while/lstmp_cell_2/Tanh_1Tanhwhile/lstmp_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstmp_cell_2/mul_2Mul while/lstmp_cell_2/Sigmoid_2:y:0while/lstmp_cell_2/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
*while/lstmp_cell_2/MatMul_2/ReadVariableOpReadVariableOp5while_lstmp_cell_2_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype0 
while/lstmp_cell_2/MatMul_2MatMulwhile_placeholder_22while/lstmp_cell_2/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstmp_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity%while/lstmp_cell_2/MatMul_2:product:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/Identity_5Identitywhile/lstmp_cell_2/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ý

while/NoOpNoOp*^while/lstmp_cell_2/BiasAdd/ReadVariableOp)^while/lstmp_cell_2/MatMul/ReadVariableOp+^while/lstmp_cell_2/MatMul_1/ReadVariableOp+^while/lstmp_cell_2/MatMul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstmp_cell_2_biasadd_readvariableop_resource4while_lstmp_cell_2_biasadd_readvariableop_resource_0"l
3while_lstmp_cell_2_matmul_1_readvariableop_resource5while_lstmp_cell_2_matmul_1_readvariableop_resource_0"l
3while_lstmp_cell_2_matmul_2_readvariableop_resource5while_lstmp_cell_2_matmul_2_readvariableop_resource_0"h
1while_lstmp_cell_2_matmul_readvariableop_resource3while_lstmp_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : 2V
)while/lstmp_cell_2/BiasAdd/ReadVariableOp)while/lstmp_cell_2/BiasAdd/ReadVariableOp2T
(while/lstmp_cell_2/MatMul/ReadVariableOp(while/lstmp_cell_2/MatMul/ReadVariableOp2X
*while/lstmp_cell_2/MatMul_1/ReadVariableOp*while/lstmp_cell_2/MatMul_1/ReadVariableOp2X
*while/lstmp_cell_2/MatMul_2/ReadVariableOp*while/lstmp_cell_2/MatMul_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
#
Ú
__inference__traced_save_3527
file_prefix-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop'
#savev2_proj_w_1_read_readvariableop:
6savev2_lstmp_2_lstmp_cell_2_kernel_read_readvariableopD
@savev2_lstmp_2_lstmp_cell_2_recurrent_kernel_read_readvariableop8
4savev2_lstmp_2_lstmp_cell_2_bias_read_readvariableop%
!savev2_proj_w_read_readvariableop:
6savev2_lstmp_3_lstmp_cell_3_kernel_read_readvariableopD
@savev2_lstmp_3_lstmp_cell_3_recurrent_kernel_read_readvariableop8
4savev2_lstmp_3_lstmp_cell_3_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ¤
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Í
valueÃBÀB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B ü
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop#savev2_proj_w_1_read_readvariableop6savev2_lstmp_2_lstmp_cell_2_kernel_read_readvariableop@savev2_lstmp_2_lstmp_cell_2_recurrent_kernel_read_readvariableop4savev2_lstmp_2_lstmp_cell_2_bias_read_readvariableop!savev2_proj_w_read_readvariableop6savev2_lstmp_3_lstmp_cell_3_kernel_read_readvariableop@savev2_lstmp_3_lstmp_cell_3_recurrent_kernel_read_readvariableop4savev2_lstmp_3_lstmp_cell_3_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*y
_input_shapesh
f: :@::@@:	:	@::@@:	@:	@:: : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:@@:%!

_output_shapes
:	:%!

_output_shapes
:	@:!

_output_shapes	
::$ 

_output_shapes

:@@:%!

_output_shapes
:	@:%	!

_output_shapes
:	@:!


_output_shapes	
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Õ2
å
 __inference__traced_restore_3573
file_prefix1
assignvariableop_dense_1_kernel:@-
assignvariableop_1_dense_1_bias:-
assignvariableop_2_proj_w_1:@@A
.assignvariableop_3_lstmp_2_lstmp_cell_2_kernel:	K
8assignvariableop_4_lstmp_2_lstmp_cell_2_recurrent_kernel:	@;
,assignvariableop_5_lstmp_2_lstmp_cell_2_bias:	+
assignvariableop_6_proj_w:@@A
.assignvariableop_7_lstmp_3_lstmp_cell_3_kernel:	@K
8assignvariableop_8_lstmp_3_lstmp_cell_3_recurrent_kernel:	@;
,assignvariableop_9_lstmp_3_lstmp_cell_3_bias:	#
assignvariableop_10_total: #
assignvariableop_11_count: 
identity_13¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9§
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Í
valueÃBÀB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B ß
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_dense_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_proj_w_1Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp.assignvariableop_3_lstmp_2_lstmp_cell_2_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_4AssignVariableOp8assignvariableop_4_lstmp_2_lstmp_cell_2_recurrent_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp,assignvariableop_5_lstmp_2_lstmp_cell_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_proj_wIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp.assignvariableop_7_lstmp_3_lstmp_cell_3_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_8AssignVariableOp8assignvariableop_8_lstmp_3_lstmp_cell_3_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp,assignvariableop_9_lstmp_3_lstmp_cell_3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ×
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_13IdentityIdentity_12:output:0^NoOp_1*
T0*
_output_shapes
: Ä
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_13Identity_13:output:0*-
_input_shapes
: : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
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
°µ
ã	
@__inference_dlstmp_layer_call_and_return_conditional_losses_2789

inputsF
3lstmp_2_lstmp_cell_2_matmul_readvariableop_resource:	H
5lstmp_2_lstmp_cell_2_matmul_1_readvariableop_resource:	@C
4lstmp_2_lstmp_cell_2_biasadd_readvariableop_resource:	G
5lstmp_2_lstmp_cell_2_matmul_2_readvariableop_resource:@@F
3lstmp_3_lstmp_cell_3_matmul_readvariableop_resource:	@H
5lstmp_3_lstmp_cell_3_matmul_1_readvariableop_resource:	@C
4lstmp_3_lstmp_cell_3_biasadd_readvariableop_resource:	G
5lstmp_3_lstmp_cell_3_matmul_2_readvariableop_resource:@@8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:
identity¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢+lstmp_2/lstmp_cell_2/BiasAdd/ReadVariableOp¢*lstmp_2/lstmp_cell_2/MatMul/ReadVariableOp¢,lstmp_2/lstmp_cell_2/MatMul_1/ReadVariableOp¢,lstmp_2/lstmp_cell_2/MatMul_2/ReadVariableOp¢lstmp_2/while¢+lstmp_3/lstmp_cell_3/BiasAdd/ReadVariableOp¢*lstmp_3/lstmp_cell_3/MatMul/ReadVariableOp¢,lstmp_3/lstmp_cell_3/MatMul_1/ReadVariableOp¢,lstmp_3/lstmp_cell_3/MatMul_2/ReadVariableOp¢lstmp_3/whileC
lstmp_2/ShapeShapeinputs*
T0*
_output_shapes
:e
lstmp_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstmp_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstmp_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
lstmp_2/strided_sliceStridedSlicelstmp_2/Shape:output:0$lstmp_2/strided_slice/stack:output:0&lstmp_2/strided_slice/stack_1:output:0&lstmp_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstmp_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@
lstmp_2/zeros/packedPacklstmp_2/strided_slice:output:0lstmp_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstmp_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstmp_2/zerosFilllstmp_2/zeros/packed:output:0lstmp_2/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
lstmp_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@
lstmp_2/zeros_1/packedPacklstmp_2/strided_slice:output:0!lstmp_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstmp_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstmp_2/zeros_1Filllstmp_2/zeros_1/packed:output:0lstmp_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
lstmp_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstmp_2/transpose	Transposeinputslstmp_2/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
lstmp_2/Shape_1Shapelstmp_2/transpose:y:0*
T0*
_output_shapes
:g
lstmp_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstmp_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstmp_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstmp_2/strided_slice_1StridedSlicelstmp_2/Shape_1:output:0&lstmp_2/strided_slice_1/stack:output:0(lstmp_2/strided_slice_1/stack_1:output:0(lstmp_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstmp_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
lstmp_2/TensorArrayV2TensorListReserve,lstmp_2/TensorArrayV2/element_shape:output:0 lstmp_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=lstmp_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ø
/lstmp_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstmp_2/transpose:y:0Flstmp_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
lstmp_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstmp_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstmp_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstmp_2/strided_slice_2StridedSlicelstmp_2/transpose:y:0&lstmp_2/strided_slice_2/stack:output:0(lstmp_2/strided_slice_2/stack_1:output:0(lstmp_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
*lstmp_2/lstmp_cell_2/MatMul/ReadVariableOpReadVariableOp3lstmp_2_lstmp_cell_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0®
lstmp_2/lstmp_cell_2/MatMulMatMul lstmp_2/strided_slice_2:output:02lstmp_2/lstmp_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
,lstmp_2/lstmp_cell_2/MatMul_1/ReadVariableOpReadVariableOp5lstmp_2_lstmp_cell_2_matmul_1_readvariableop_resource*
_output_shapes
:	@*
dtype0¨
lstmp_2/lstmp_cell_2/MatMul_1MatMullstmp_2/zeros:output:04lstmp_2/lstmp_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
lstmp_2/lstmp_cell_2/addAddV2%lstmp_2/lstmp_cell_2/MatMul:product:0'lstmp_2/lstmp_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+lstmp_2/lstmp_cell_2/BiasAdd/ReadVariableOpReadVariableOp4lstmp_2_lstmp_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
lstmp_2/lstmp_cell_2/BiasAddBiasAddlstmp_2/lstmp_cell_2/add:z:03lstmp_2/lstmp_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$lstmp_2/lstmp_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :õ
lstmp_2/lstmp_cell_2/splitSplit-lstmp_2/lstmp_cell_2/split/split_dim:output:0%lstmp_2/lstmp_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*
	num_split~
lstmp_2/lstmp_cell_2/SigmoidSigmoid#lstmp_2/lstmp_cell_2/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_2/lstmp_cell_2/Sigmoid_1Sigmoid#lstmp_2/lstmp_cell_2/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_2/lstmp_cell_2/mulMul"lstmp_2/lstmp_cell_2/Sigmoid_1:y:0lstmp_2/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
lstmp_2/lstmp_cell_2/TanhTanh#lstmp_2/lstmp_cell_2/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_2/lstmp_cell_2/mul_1Mul lstmp_2/lstmp_cell_2/Sigmoid:y:0lstmp_2/lstmp_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_2/lstmp_cell_2/add_1AddV2lstmp_2/lstmp_cell_2/mul:z:0lstmp_2/lstmp_cell_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_2/lstmp_cell_2/Sigmoid_2Sigmoid#lstmp_2/lstmp_cell_2/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@u
lstmp_2/lstmp_cell_2/Tanh_1Tanhlstmp_2/lstmp_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_2/lstmp_cell_2/mul_2Mul"lstmp_2/lstmp_cell_2/Sigmoid_2:y:0lstmp_2/lstmp_cell_2/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
,lstmp_2/lstmp_cell_2/MatMul_2/ReadVariableOpReadVariableOp5lstmp_2_lstmp_cell_2_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype0§
lstmp_2/lstmp_cell_2/MatMul_2MatMullstmp_2/zeros:output:04lstmp_2/lstmp_cell_2/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@v
%lstmp_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ð
lstmp_2/TensorArrayV2_1TensorListReserve.lstmp_2/TensorArrayV2_1/element_shape:output:0 lstmp_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
lstmp_2/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstmp_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
lstmp_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : «
lstmp_2/whileWhile#lstmp_2/while/loop_counter:output:0)lstmp_2/while/maximum_iterations:output:0lstmp_2/time:output:0 lstmp_2/TensorArrayV2_1:handle:0lstmp_2/zeros:output:0lstmp_2/zeros_1:output:0 lstmp_2/strided_slice_1:output:0?lstmp_2/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstmp_2_lstmp_cell_2_matmul_readvariableop_resource5lstmp_2_lstmp_cell_2_matmul_1_readvariableop_resource4lstmp_2_lstmp_cell_2_biasadd_readvariableop_resource5lstmp_2_lstmp_cell_2_matmul_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *&
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
lstmp_2_while_body_2547*#
condR
lstmp_2_while_cond_2546*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *
parallel_iterations 
8lstmp_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ú
*lstmp_2/TensorArrayV2Stack/TensorListStackTensorListStacklstmp_2/while:output:3Alstmp_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0p
lstmp_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
lstmp_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstmp_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
lstmp_2/strided_slice_3StridedSlice3lstmp_2/TensorArrayV2Stack/TensorListStack:tensor:0&lstmp_2/strided_slice_3/stack:output:0(lstmp_2/strided_slice_3/stack_1:output:0(lstmp_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maskm
lstmp_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ®
lstmp_2/transpose_1	Transpose3lstmp_2/TensorArrayV2Stack/TensorListStack:tensor:0!lstmp_2/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@T
lstmp_3/ShapeShapelstmp_2/transpose_1:y:0*
T0*
_output_shapes
:e
lstmp_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstmp_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstmp_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
lstmp_3/strided_sliceStridedSlicelstmp_3/Shape:output:0$lstmp_3/strided_slice/stack:output:0&lstmp_3/strided_slice/stack_1:output:0&lstmp_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstmp_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@
lstmp_3/zeros/packedPacklstmp_3/strided_slice:output:0lstmp_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstmp_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstmp_3/zerosFilllstmp_3/zeros/packed:output:0lstmp_3/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
lstmp_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@
lstmp_3/zeros_1/packedPacklstmp_3/strided_slice:output:0!lstmp_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstmp_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstmp_3/zeros_1Filllstmp_3/zeros_1/packed:output:0lstmp_3/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
lstmp_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstmp_3/transpose	Transposelstmp_2/transpose_1:y:0lstmp_3/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@T
lstmp_3/Shape_1Shapelstmp_3/transpose:y:0*
T0*
_output_shapes
:g
lstmp_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstmp_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstmp_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstmp_3/strided_slice_1StridedSlicelstmp_3/Shape_1:output:0&lstmp_3/strided_slice_1/stack:output:0(lstmp_3/strided_slice_1/stack_1:output:0(lstmp_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstmp_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
lstmp_3/TensorArrayV2TensorListReserve,lstmp_3/TensorArrayV2/element_shape:output:0 lstmp_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=lstmp_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ø
/lstmp_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstmp_3/transpose:y:0Flstmp_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
lstmp_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstmp_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstmp_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstmp_3/strided_slice_2StridedSlicelstmp_3/transpose:y:0&lstmp_3/strided_slice_2/stack:output:0(lstmp_3/strided_slice_2/stack_1:output:0(lstmp_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
*lstmp_3/lstmp_cell_3/MatMul/ReadVariableOpReadVariableOp3lstmp_3_lstmp_cell_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0®
lstmp_3/lstmp_cell_3/MatMulMatMul lstmp_3/strided_slice_2:output:02lstmp_3/lstmp_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
,lstmp_3/lstmp_cell_3/MatMul_1/ReadVariableOpReadVariableOp5lstmp_3_lstmp_cell_3_matmul_1_readvariableop_resource*
_output_shapes
:	@*
dtype0¨
lstmp_3/lstmp_cell_3/MatMul_1MatMullstmp_3/zeros:output:04lstmp_3/lstmp_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
lstmp_3/lstmp_cell_3/addAddV2%lstmp_3/lstmp_cell_3/MatMul:product:0'lstmp_3/lstmp_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+lstmp_3/lstmp_cell_3/BiasAdd/ReadVariableOpReadVariableOp4lstmp_3_lstmp_cell_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
lstmp_3/lstmp_cell_3/BiasAddBiasAddlstmp_3/lstmp_cell_3/add:z:03lstmp_3/lstmp_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$lstmp_3/lstmp_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :õ
lstmp_3/lstmp_cell_3/splitSplit-lstmp_3/lstmp_cell_3/split/split_dim:output:0%lstmp_3/lstmp_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*
	num_split~
lstmp_3/lstmp_cell_3/SigmoidSigmoid#lstmp_3/lstmp_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_3/lstmp_cell_3/Sigmoid_1Sigmoid#lstmp_3/lstmp_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_3/lstmp_cell_3/mulMul"lstmp_3/lstmp_cell_3/Sigmoid_1:y:0lstmp_3/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
lstmp_3/lstmp_cell_3/TanhTanh#lstmp_3/lstmp_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_3/lstmp_cell_3/mul_1Mul lstmp_3/lstmp_cell_3/Sigmoid:y:0lstmp_3/lstmp_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_3/lstmp_cell_3/add_1AddV2lstmp_3/lstmp_cell_3/mul:z:0lstmp_3/lstmp_cell_3/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_3/lstmp_cell_3/Sigmoid_2Sigmoid#lstmp_3/lstmp_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@u
lstmp_3/lstmp_cell_3/Tanh_1Tanhlstmp_3/lstmp_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_3/lstmp_cell_3/mul_2Mul"lstmp_3/lstmp_cell_3/Sigmoid_2:y:0lstmp_3/lstmp_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
,lstmp_3/lstmp_cell_3/MatMul_2/ReadVariableOpReadVariableOp5lstmp_3_lstmp_cell_3_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype0§
lstmp_3/lstmp_cell_3/MatMul_2MatMullstmp_3/zeros:output:04lstmp_3/lstmp_cell_3/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@v
%lstmp_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   f
$lstmp_3/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstmp_3/TensorArrayV2_1TensorListReserve.lstmp_3/TensorArrayV2_1/element_shape:output:0-lstmp_3/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
lstmp_3/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstmp_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
lstmp_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : «
lstmp_3/whileWhile#lstmp_3/while/loop_counter:output:0)lstmp_3/while/maximum_iterations:output:0lstmp_3/time:output:0 lstmp_3/TensorArrayV2_1:handle:0lstmp_3/zeros:output:0lstmp_3/zeros_1:output:0 lstmp_3/strided_slice_1:output:0?lstmp_3/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstmp_3_lstmp_cell_3_matmul_readvariableop_resource5lstmp_3_lstmp_cell_3_matmul_1_readvariableop_resource4lstmp_3_lstmp_cell_3_biasadd_readvariableop_resource5lstmp_3_lstmp_cell_3_matmul_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *&
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
lstmp_3_while_body_2694*#
condR
lstmp_3_while_cond_2693*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *
parallel_iterations 
8lstmp_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   î
*lstmp_3/TensorArrayV2Stack/TensorListStackTensorListStacklstmp_3/while:output:3Alstmp_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elementsp
lstmp_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
lstmp_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstmp_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
lstmp_3/strided_slice_3StridedSlice3lstmp_3/TensorArrayV2Stack/TensorListStack:tensor:0&lstmp_3/strided_slice_3/stack:output:0(lstmp_3/strided_slice_3/stack_1:output:0(lstmp_3/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maskm
lstmp_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ®
lstmp_3/transpose_1	Transpose3lstmp_3/TensorArrayV2Stack/TensorListStack:tensor:0!lstmp_3/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_1/MatMulMatMul lstmp_3/strided_slice_3:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp,^lstmp_2/lstmp_cell_2/BiasAdd/ReadVariableOp+^lstmp_2/lstmp_cell_2/MatMul/ReadVariableOp-^lstmp_2/lstmp_cell_2/MatMul_1/ReadVariableOp-^lstmp_2/lstmp_cell_2/MatMul_2/ReadVariableOp^lstmp_2/while,^lstmp_3/lstmp_cell_3/BiasAdd/ReadVariableOp+^lstmp_3/lstmp_cell_3/MatMul/ReadVariableOp-^lstmp_3/lstmp_cell_3/MatMul_1/ReadVariableOp-^lstmp_3/lstmp_cell_3/MatMul_2/ReadVariableOp^lstmp_3/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2Z
+lstmp_2/lstmp_cell_2/BiasAdd/ReadVariableOp+lstmp_2/lstmp_cell_2/BiasAdd/ReadVariableOp2X
*lstmp_2/lstmp_cell_2/MatMul/ReadVariableOp*lstmp_2/lstmp_cell_2/MatMul/ReadVariableOp2\
,lstmp_2/lstmp_cell_2/MatMul_1/ReadVariableOp,lstmp_2/lstmp_cell_2/MatMul_1/ReadVariableOp2\
,lstmp_2/lstmp_cell_2/MatMul_2/ReadVariableOp,lstmp_2/lstmp_cell_2/MatMul_2/ReadVariableOp2
lstmp_2/whilelstmp_2/while2Z
+lstmp_3/lstmp_cell_3/BiasAdd/ReadVariableOp+lstmp_3/lstmp_cell_3/BiasAdd/ReadVariableOp2X
*lstmp_3/lstmp_cell_3/MatMul/ReadVariableOp*lstmp_3/lstmp_cell_3/MatMul/ReadVariableOp2\
,lstmp_3/lstmp_cell_3/MatMul_1/ReadVariableOp,lstmp_3/lstmp_cell_3/MatMul_1/ReadVariableOp2\
,lstmp_3/lstmp_cell_3/MatMul_2/ReadVariableOp,lstmp_3/lstmp_cell_3/MatMul_2/ReadVariableOp2
lstmp_3/whilelstmp_3/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾I
£
lstmp_3_while_body_2390,
(lstmp_3_while_lstmp_3_while_loop_counter2
.lstmp_3_while_lstmp_3_while_maximum_iterations
lstmp_3_while_placeholder
lstmp_3_while_placeholder_1
lstmp_3_while_placeholder_2
lstmp_3_while_placeholder_3+
'lstmp_3_while_lstmp_3_strided_slice_1_0g
clstmp_3_while_tensorarrayv2read_tensorlistgetitem_lstmp_3_tensorarrayunstack_tensorlistfromtensor_0N
;lstmp_3_while_lstmp_cell_3_matmul_readvariableop_resource_0:	@P
=lstmp_3_while_lstmp_cell_3_matmul_1_readvariableop_resource_0:	@K
<lstmp_3_while_lstmp_cell_3_biasadd_readvariableop_resource_0:	O
=lstmp_3_while_lstmp_cell_3_matmul_2_readvariableop_resource_0:@@
lstmp_3_while_identity
lstmp_3_while_identity_1
lstmp_3_while_identity_2
lstmp_3_while_identity_3
lstmp_3_while_identity_4
lstmp_3_while_identity_5)
%lstmp_3_while_lstmp_3_strided_slice_1e
alstmp_3_while_tensorarrayv2read_tensorlistgetitem_lstmp_3_tensorarrayunstack_tensorlistfromtensorL
9lstmp_3_while_lstmp_cell_3_matmul_readvariableop_resource:	@N
;lstmp_3_while_lstmp_cell_3_matmul_1_readvariableop_resource:	@I
:lstmp_3_while_lstmp_cell_3_biasadd_readvariableop_resource:	M
;lstmp_3_while_lstmp_cell_3_matmul_2_readvariableop_resource:@@¢1lstmp_3/while/lstmp_cell_3/BiasAdd/ReadVariableOp¢0lstmp_3/while/lstmp_cell_3/MatMul/ReadVariableOp¢2lstmp_3/while/lstmp_cell_3/MatMul_1/ReadVariableOp¢2lstmp_3/while/lstmp_cell_3/MatMul_2/ReadVariableOp
?lstmp_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Î
1lstmp_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstmp_3_while_tensorarrayv2read_tensorlistgetitem_lstmp_3_tensorarrayunstack_tensorlistfromtensor_0lstmp_3_while_placeholderHlstmp_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0­
0lstmp_3/while/lstmp_cell_3/MatMul/ReadVariableOpReadVariableOp;lstmp_3_while_lstmp_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	@*
dtype0Ò
!lstmp_3/while/lstmp_cell_3/MatMulMatMul8lstmp_3/while/TensorArrayV2Read/TensorListGetItem:item:08lstmp_3/while/lstmp_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
2lstmp_3/while/lstmp_cell_3/MatMul_1/ReadVariableOpReadVariableOp=lstmp_3_while_lstmp_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes
:	@*
dtype0¹
#lstmp_3/while/lstmp_cell_3/MatMul_1MatMullstmp_3_while_placeholder_2:lstmp_3/while/lstmp_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
lstmp_3/while/lstmp_cell_3/addAddV2+lstmp_3/while/lstmp_cell_3/MatMul:product:0-lstmp_3/while/lstmp_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
1lstmp_3/while/lstmp_cell_3/BiasAdd/ReadVariableOpReadVariableOp<lstmp_3_while_lstmp_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0¿
"lstmp_3/while/lstmp_cell_3/BiasAddBiasAdd"lstmp_3/while/lstmp_cell_3/add:z:09lstmp_3/while/lstmp_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
*lstmp_3/while/lstmp_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstmp_3/while/lstmp_cell_3/splitSplit3lstmp_3/while/lstmp_cell_3/split/split_dim:output:0+lstmp_3/while/lstmp_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*
	num_split
"lstmp_3/while/lstmp_cell_3/SigmoidSigmoid)lstmp_3/while/lstmp_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
$lstmp_3/while/lstmp_cell_3/Sigmoid_1Sigmoid)lstmp_3/while/lstmp_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_3/while/lstmp_cell_3/mulMul(lstmp_3/while/lstmp_cell_3/Sigmoid_1:y:0lstmp_3_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_3/while/lstmp_cell_3/TanhTanh)lstmp_3/while/lstmp_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
 lstmp_3/while/lstmp_cell_3/mul_1Mul&lstmp_3/while/lstmp_cell_3/Sigmoid:y:0#lstmp_3/while/lstmp_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
 lstmp_3/while/lstmp_cell_3/add_1AddV2"lstmp_3/while/lstmp_cell_3/mul:z:0$lstmp_3/while/lstmp_cell_3/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
$lstmp_3/while/lstmp_cell_3/Sigmoid_2Sigmoid)lstmp_3/while/lstmp_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
!lstmp_3/while/lstmp_cell_3/Tanh_1Tanh$lstmp_3/while/lstmp_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ª
 lstmp_3/while/lstmp_cell_3/mul_2Mul(lstmp_3/while/lstmp_cell_3/Sigmoid_2:y:0%lstmp_3/while/lstmp_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@°
2lstmp_3/while/lstmp_cell_3/MatMul_2/ReadVariableOpReadVariableOp=lstmp_3_while_lstmp_cell_3_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¸
#lstmp_3/while/lstmp_cell_3/MatMul_2MatMullstmp_3_while_placeholder_2:lstmp_3/while/lstmp_cell_3/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
8lstmp_3/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
2lstmp_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstmp_3_while_placeholder_1Alstmp_3/while/TensorArrayV2Write/TensorListSetItem/index:output:0$lstmp_3/while/lstmp_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒU
lstmp_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstmp_3/while/addAddV2lstmp_3_while_placeholderlstmp_3/while/add/y:output:0*
T0*
_output_shapes
: W
lstmp_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstmp_3/while/add_1AddV2(lstmp_3_while_lstmp_3_while_loop_counterlstmp_3/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstmp_3/while/IdentityIdentitylstmp_3/while/add_1:z:0^lstmp_3/while/NoOp*
T0*
_output_shapes
: 
lstmp_3/while/Identity_1Identity.lstmp_3_while_lstmp_3_while_maximum_iterations^lstmp_3/while/NoOp*
T0*
_output_shapes
: q
lstmp_3/while/Identity_2Identitylstmp_3/while/add:z:0^lstmp_3/while/NoOp*
T0*
_output_shapes
: 
lstmp_3/while/Identity_3IdentityBlstmp_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstmp_3/while/NoOp*
T0*
_output_shapes
: 
lstmp_3/while/Identity_4Identity-lstmp_3/while/lstmp_cell_3/MatMul_2:product:0^lstmp_3/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_3/while/Identity_5Identity$lstmp_3/while/lstmp_cell_3/add_1:z:0^lstmp_3/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
lstmp_3/while/NoOpNoOp2^lstmp_3/while/lstmp_cell_3/BiasAdd/ReadVariableOp1^lstmp_3/while/lstmp_cell_3/MatMul/ReadVariableOp3^lstmp_3/while/lstmp_cell_3/MatMul_1/ReadVariableOp3^lstmp_3/while/lstmp_cell_3/MatMul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstmp_3_while_identitylstmp_3/while/Identity:output:0"=
lstmp_3_while_identity_1!lstmp_3/while/Identity_1:output:0"=
lstmp_3_while_identity_2!lstmp_3/while/Identity_2:output:0"=
lstmp_3_while_identity_3!lstmp_3/while/Identity_3:output:0"=
lstmp_3_while_identity_4!lstmp_3/while/Identity_4:output:0"=
lstmp_3_while_identity_5!lstmp_3/while/Identity_5:output:0"P
%lstmp_3_while_lstmp_3_strided_slice_1'lstmp_3_while_lstmp_3_strided_slice_1_0"z
:lstmp_3_while_lstmp_cell_3_biasadd_readvariableop_resource<lstmp_3_while_lstmp_cell_3_biasadd_readvariableop_resource_0"|
;lstmp_3_while_lstmp_cell_3_matmul_1_readvariableop_resource=lstmp_3_while_lstmp_cell_3_matmul_1_readvariableop_resource_0"|
;lstmp_3_while_lstmp_cell_3_matmul_2_readvariableop_resource=lstmp_3_while_lstmp_cell_3_matmul_2_readvariableop_resource_0"x
9lstmp_3_while_lstmp_cell_3_matmul_readvariableop_resource;lstmp_3_while_lstmp_cell_3_matmul_readvariableop_resource_0"È
alstmp_3_while_tensorarrayv2read_tensorlistgetitem_lstmp_3_tensorarrayunstack_tensorlistfromtensorclstmp_3_while_tensorarrayv2read_tensorlistgetitem_lstmp_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : 2f
1lstmp_3/while/lstmp_cell_3/BiasAdd/ReadVariableOp1lstmp_3/while/lstmp_cell_3/BiasAdd/ReadVariableOp2d
0lstmp_3/while/lstmp_cell_3/MatMul/ReadVariableOp0lstmp_3/while/lstmp_cell_3/MatMul/ReadVariableOp2h
2lstmp_3/while/lstmp_cell_3/MatMul_1/ReadVariableOp2lstmp_3/while/lstmp_cell_3/MatMul_1/ReadVariableOp2h
2lstmp_3/while/lstmp_cell_3/MatMul_2/ReadVariableOp2lstmp_3/while/lstmp_cell_3/MatMul_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
ìN

A__inference_lstmp_3_layer_call_and_return_conditional_losses_3295

inputs>
+lstmp_cell_3_matmul_readvariableop_resource:	@@
-lstmp_cell_3_matmul_1_readvariableop_resource:	@;
,lstmp_cell_3_biasadd_readvariableop_resource:	?
-lstmp_cell_3_matmul_2_readvariableop_resource:@@
identity¢#lstmp_cell_3/BiasAdd/ReadVariableOp¢"lstmp_cell_3/MatMul/ReadVariableOp¢$lstmp_cell_3/MatMul_1/ReadVariableOp¢$lstmp_cell_3/MatMul_2/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
"lstmp_cell_3/MatMul/ReadVariableOpReadVariableOp+lstmp_cell_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
lstmp_cell_3/MatMulMatMulstrided_slice_2:output:0*lstmp_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstmp_cell_3/MatMul_1/ReadVariableOpReadVariableOp-lstmp_cell_3_matmul_1_readvariableop_resource*
_output_shapes
:	@*
dtype0
lstmp_cell_3/MatMul_1MatMulzeros:output:0,lstmp_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstmp_cell_3/addAddV2lstmp_cell_3/MatMul:product:0lstmp_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstmp_cell_3/BiasAdd/ReadVariableOpReadVariableOp,lstmp_cell_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstmp_cell_3/BiasAddBiasAddlstmp_cell_3/add:z:0+lstmp_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstmp_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstmp_cell_3/splitSplit%lstmp_cell_3/split/split_dim:output:0lstmp_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*
	num_splitn
lstmp_cell_3/SigmoidSigmoidlstmp_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@p
lstmp_cell_3/Sigmoid_1Sigmoidlstmp_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
lstmp_cell_3/mulMullstmp_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
lstmp_cell_3/TanhTanhlstmp_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
lstmp_cell_3/mul_1Mullstmp_cell_3/Sigmoid:y:0lstmp_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
lstmp_cell_3/add_1AddV2lstmp_cell_3/mul:z:0lstmp_cell_3/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@p
lstmp_cell_3/Sigmoid_2Sigmoidlstmp_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
lstmp_cell_3/Tanh_1Tanhlstmp_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_cell_3/mul_2Mullstmp_cell_3/Sigmoid_2:y:0lstmp_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
$lstmp_cell_3/MatMul_2/ReadVariableOpReadVariableOp-lstmp_cell_3_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype0
lstmp_cell_3/MatMul_2MatMulzeros:output:0,lstmp_cell_3/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ³
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstmp_cell_3_matmul_readvariableop_resource-lstmp_cell_3_matmul_1_readvariableop_resource,lstmp_cell_3_biasadd_readvariableop_resource-lstmp_cell_3_matmul_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *&
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3206*
condR
while_cond_3205*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ç
NoOpNoOp$^lstmp_cell_3/BiasAdd/ReadVariableOp#^lstmp_cell_3/MatMul/ReadVariableOp%^lstmp_cell_3/MatMul_1/ReadVariableOp%^lstmp_cell_3/MatMul_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : : : 2J
#lstmp_cell_3/BiasAdd/ReadVariableOp#lstmp_cell_3/BiasAdd/ReadVariableOp2H
"lstmp_cell_3/MatMul/ReadVariableOp"lstmp_cell_3/MatMul/ReadVariableOp2L
$lstmp_cell_3/MatMul_1/ReadVariableOp$lstmp_cell_3/MatMul_1/ReadVariableOp2L
$lstmp_cell_3/MatMul_2/ReadVariableOp$lstmp_cell_3/MatMul_2/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ä	
ò
A__inference_dense_1_layer_call_and_return_conditional_losses_3468

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ã
í
while_cond_1430
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_12
.while_while_cond_1430___redundant_placeholder02
.while_while_cond_1430___redundant_placeholder12
.while_while_cond_1430___redundant_placeholder22
.while_while_cond_1430___redundant_placeholder32
.while_while_cond_1430___redundant_placeholder4
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
æM

A__inference_lstmp_2_layer_call_and_return_conditional_losses_3115

inputs>
+lstmp_cell_2_matmul_readvariableop_resource:	@
-lstmp_cell_2_matmul_1_readvariableop_resource:	@;
,lstmp_cell_2_biasadd_readvariableop_resource:	?
-lstmp_cell_2_matmul_2_readvariableop_resource:@@
identity¢#lstmp_cell_2/BiasAdd/ReadVariableOp¢"lstmp_cell_2/MatMul/ReadVariableOp¢$lstmp_cell_2/MatMul_1/ReadVariableOp¢$lstmp_cell_2/MatMul_2/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstmp_cell_2/MatMul/ReadVariableOpReadVariableOp+lstmp_cell_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstmp_cell_2/MatMulMatMulstrided_slice_2:output:0*lstmp_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstmp_cell_2/MatMul_1/ReadVariableOpReadVariableOp-lstmp_cell_2_matmul_1_readvariableop_resource*
_output_shapes
:	@*
dtype0
lstmp_cell_2/MatMul_1MatMulzeros:output:0,lstmp_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstmp_cell_2/addAddV2lstmp_cell_2/MatMul:product:0lstmp_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstmp_cell_2/BiasAdd/ReadVariableOpReadVariableOp,lstmp_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstmp_cell_2/BiasAddBiasAddlstmp_cell_2/add:z:0+lstmp_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstmp_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstmp_cell_2/splitSplit%lstmp_cell_2/split/split_dim:output:0lstmp_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*
	num_splitn
lstmp_cell_2/SigmoidSigmoidlstmp_cell_2/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@p
lstmp_cell_2/Sigmoid_1Sigmoidlstmp_cell_2/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
lstmp_cell_2/mulMullstmp_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
lstmp_cell_2/TanhTanhlstmp_cell_2/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
lstmp_cell_2/mul_1Mullstmp_cell_2/Sigmoid:y:0lstmp_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
lstmp_cell_2/add_1AddV2lstmp_cell_2/mul:z:0lstmp_cell_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@p
lstmp_cell_2/Sigmoid_2Sigmoidlstmp_cell_2/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
lstmp_cell_2/Tanh_1Tanhlstmp_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_cell_2/mul_2Mullstmp_cell_2/Sigmoid_2:y:0lstmp_cell_2/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
$lstmp_cell_2/MatMul_2/ReadVariableOpReadVariableOp-lstmp_cell_2_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype0
lstmp_cell_2/MatMul_2MatMulzeros:output:0,lstmp_cell_2/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ³
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstmp_cell_2_matmul_readvariableop_resource-lstmp_cell_2_matmul_1_readvariableop_resource,lstmp_cell_2_biasadd_readvariableop_resource-lstmp_cell_2_matmul_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *&
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3027*
condR
while_cond_3026*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ç
NoOpNoOp$^lstmp_cell_2/BiasAdd/ReadVariableOp#^lstmp_cell_2/MatMul/ReadVariableOp%^lstmp_cell_2/MatMul_1/ReadVariableOp%^lstmp_cell_2/MatMul_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : 2J
#lstmp_cell_2/BiasAdd/ReadVariableOp#lstmp_cell_2/BiasAdd/ReadVariableOp2H
"lstmp_cell_2/MatMul/ReadVariableOp"lstmp_cell_2/MatMul/ReadVariableOp2L
$lstmp_cell_2/MatMul_1/ReadVariableOp$lstmp_cell_2/MatMul_1/ReadVariableOp2L
$lstmp_cell_2/MatMul_2/ReadVariableOp$lstmp_cell_2/MatMul_2/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä	
ò
A__inference_dense_1_layer_call_and_return_conditional_losses_1540

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
°µ
ã	
@__inference_dlstmp_layer_call_and_return_conditional_losses_2485

inputsF
3lstmp_2_lstmp_cell_2_matmul_readvariableop_resource:	H
5lstmp_2_lstmp_cell_2_matmul_1_readvariableop_resource:	@C
4lstmp_2_lstmp_cell_2_biasadd_readvariableop_resource:	G
5lstmp_2_lstmp_cell_2_matmul_2_readvariableop_resource:@@F
3lstmp_3_lstmp_cell_3_matmul_readvariableop_resource:	@H
5lstmp_3_lstmp_cell_3_matmul_1_readvariableop_resource:	@C
4lstmp_3_lstmp_cell_3_biasadd_readvariableop_resource:	G
5lstmp_3_lstmp_cell_3_matmul_2_readvariableop_resource:@@8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:
identity¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢+lstmp_2/lstmp_cell_2/BiasAdd/ReadVariableOp¢*lstmp_2/lstmp_cell_2/MatMul/ReadVariableOp¢,lstmp_2/lstmp_cell_2/MatMul_1/ReadVariableOp¢,lstmp_2/lstmp_cell_2/MatMul_2/ReadVariableOp¢lstmp_2/while¢+lstmp_3/lstmp_cell_3/BiasAdd/ReadVariableOp¢*lstmp_3/lstmp_cell_3/MatMul/ReadVariableOp¢,lstmp_3/lstmp_cell_3/MatMul_1/ReadVariableOp¢,lstmp_3/lstmp_cell_3/MatMul_2/ReadVariableOp¢lstmp_3/whileC
lstmp_2/ShapeShapeinputs*
T0*
_output_shapes
:e
lstmp_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstmp_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstmp_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
lstmp_2/strided_sliceStridedSlicelstmp_2/Shape:output:0$lstmp_2/strided_slice/stack:output:0&lstmp_2/strided_slice/stack_1:output:0&lstmp_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstmp_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@
lstmp_2/zeros/packedPacklstmp_2/strided_slice:output:0lstmp_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstmp_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstmp_2/zerosFilllstmp_2/zeros/packed:output:0lstmp_2/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
lstmp_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@
lstmp_2/zeros_1/packedPacklstmp_2/strided_slice:output:0!lstmp_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstmp_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstmp_2/zeros_1Filllstmp_2/zeros_1/packed:output:0lstmp_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
lstmp_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstmp_2/transpose	Transposeinputslstmp_2/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
lstmp_2/Shape_1Shapelstmp_2/transpose:y:0*
T0*
_output_shapes
:g
lstmp_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstmp_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstmp_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstmp_2/strided_slice_1StridedSlicelstmp_2/Shape_1:output:0&lstmp_2/strided_slice_1/stack:output:0(lstmp_2/strided_slice_1/stack_1:output:0(lstmp_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstmp_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
lstmp_2/TensorArrayV2TensorListReserve,lstmp_2/TensorArrayV2/element_shape:output:0 lstmp_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=lstmp_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ø
/lstmp_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstmp_2/transpose:y:0Flstmp_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
lstmp_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstmp_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstmp_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstmp_2/strided_slice_2StridedSlicelstmp_2/transpose:y:0&lstmp_2/strided_slice_2/stack:output:0(lstmp_2/strided_slice_2/stack_1:output:0(lstmp_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
*lstmp_2/lstmp_cell_2/MatMul/ReadVariableOpReadVariableOp3lstmp_2_lstmp_cell_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0®
lstmp_2/lstmp_cell_2/MatMulMatMul lstmp_2/strided_slice_2:output:02lstmp_2/lstmp_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
,lstmp_2/lstmp_cell_2/MatMul_1/ReadVariableOpReadVariableOp5lstmp_2_lstmp_cell_2_matmul_1_readvariableop_resource*
_output_shapes
:	@*
dtype0¨
lstmp_2/lstmp_cell_2/MatMul_1MatMullstmp_2/zeros:output:04lstmp_2/lstmp_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
lstmp_2/lstmp_cell_2/addAddV2%lstmp_2/lstmp_cell_2/MatMul:product:0'lstmp_2/lstmp_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+lstmp_2/lstmp_cell_2/BiasAdd/ReadVariableOpReadVariableOp4lstmp_2_lstmp_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
lstmp_2/lstmp_cell_2/BiasAddBiasAddlstmp_2/lstmp_cell_2/add:z:03lstmp_2/lstmp_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$lstmp_2/lstmp_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :õ
lstmp_2/lstmp_cell_2/splitSplit-lstmp_2/lstmp_cell_2/split/split_dim:output:0%lstmp_2/lstmp_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*
	num_split~
lstmp_2/lstmp_cell_2/SigmoidSigmoid#lstmp_2/lstmp_cell_2/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_2/lstmp_cell_2/Sigmoid_1Sigmoid#lstmp_2/lstmp_cell_2/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_2/lstmp_cell_2/mulMul"lstmp_2/lstmp_cell_2/Sigmoid_1:y:0lstmp_2/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
lstmp_2/lstmp_cell_2/TanhTanh#lstmp_2/lstmp_cell_2/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_2/lstmp_cell_2/mul_1Mul lstmp_2/lstmp_cell_2/Sigmoid:y:0lstmp_2/lstmp_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_2/lstmp_cell_2/add_1AddV2lstmp_2/lstmp_cell_2/mul:z:0lstmp_2/lstmp_cell_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_2/lstmp_cell_2/Sigmoid_2Sigmoid#lstmp_2/lstmp_cell_2/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@u
lstmp_2/lstmp_cell_2/Tanh_1Tanhlstmp_2/lstmp_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_2/lstmp_cell_2/mul_2Mul"lstmp_2/lstmp_cell_2/Sigmoid_2:y:0lstmp_2/lstmp_cell_2/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
,lstmp_2/lstmp_cell_2/MatMul_2/ReadVariableOpReadVariableOp5lstmp_2_lstmp_cell_2_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype0§
lstmp_2/lstmp_cell_2/MatMul_2MatMullstmp_2/zeros:output:04lstmp_2/lstmp_cell_2/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@v
%lstmp_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ð
lstmp_2/TensorArrayV2_1TensorListReserve.lstmp_2/TensorArrayV2_1/element_shape:output:0 lstmp_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
lstmp_2/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstmp_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
lstmp_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : «
lstmp_2/whileWhile#lstmp_2/while/loop_counter:output:0)lstmp_2/while/maximum_iterations:output:0lstmp_2/time:output:0 lstmp_2/TensorArrayV2_1:handle:0lstmp_2/zeros:output:0lstmp_2/zeros_1:output:0 lstmp_2/strided_slice_1:output:0?lstmp_2/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstmp_2_lstmp_cell_2_matmul_readvariableop_resource5lstmp_2_lstmp_cell_2_matmul_1_readvariableop_resource4lstmp_2_lstmp_cell_2_biasadd_readvariableop_resource5lstmp_2_lstmp_cell_2_matmul_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *&
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
lstmp_2_while_body_2243*#
condR
lstmp_2_while_cond_2242*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *
parallel_iterations 
8lstmp_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ú
*lstmp_2/TensorArrayV2Stack/TensorListStackTensorListStacklstmp_2/while:output:3Alstmp_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0p
lstmp_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
lstmp_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstmp_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
lstmp_2/strided_slice_3StridedSlice3lstmp_2/TensorArrayV2Stack/TensorListStack:tensor:0&lstmp_2/strided_slice_3/stack:output:0(lstmp_2/strided_slice_3/stack_1:output:0(lstmp_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maskm
lstmp_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ®
lstmp_2/transpose_1	Transpose3lstmp_2/TensorArrayV2Stack/TensorListStack:tensor:0!lstmp_2/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@T
lstmp_3/ShapeShapelstmp_2/transpose_1:y:0*
T0*
_output_shapes
:e
lstmp_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstmp_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstmp_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
lstmp_3/strided_sliceStridedSlicelstmp_3/Shape:output:0$lstmp_3/strided_slice/stack:output:0&lstmp_3/strided_slice/stack_1:output:0&lstmp_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstmp_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@
lstmp_3/zeros/packedPacklstmp_3/strided_slice:output:0lstmp_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstmp_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstmp_3/zerosFilllstmp_3/zeros/packed:output:0lstmp_3/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
lstmp_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@
lstmp_3/zeros_1/packedPacklstmp_3/strided_slice:output:0!lstmp_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstmp_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstmp_3/zeros_1Filllstmp_3/zeros_1/packed:output:0lstmp_3/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
lstmp_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstmp_3/transpose	Transposelstmp_2/transpose_1:y:0lstmp_3/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@T
lstmp_3/Shape_1Shapelstmp_3/transpose:y:0*
T0*
_output_shapes
:g
lstmp_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstmp_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstmp_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstmp_3/strided_slice_1StridedSlicelstmp_3/Shape_1:output:0&lstmp_3/strided_slice_1/stack:output:0(lstmp_3/strided_slice_1/stack_1:output:0(lstmp_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstmp_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
lstmp_3/TensorArrayV2TensorListReserve,lstmp_3/TensorArrayV2/element_shape:output:0 lstmp_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=lstmp_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ø
/lstmp_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstmp_3/transpose:y:0Flstmp_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
lstmp_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstmp_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstmp_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstmp_3/strided_slice_2StridedSlicelstmp_3/transpose:y:0&lstmp_3/strided_slice_2/stack:output:0(lstmp_3/strided_slice_2/stack_1:output:0(lstmp_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
*lstmp_3/lstmp_cell_3/MatMul/ReadVariableOpReadVariableOp3lstmp_3_lstmp_cell_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0®
lstmp_3/lstmp_cell_3/MatMulMatMul lstmp_3/strided_slice_2:output:02lstmp_3/lstmp_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
,lstmp_3/lstmp_cell_3/MatMul_1/ReadVariableOpReadVariableOp5lstmp_3_lstmp_cell_3_matmul_1_readvariableop_resource*
_output_shapes
:	@*
dtype0¨
lstmp_3/lstmp_cell_3/MatMul_1MatMullstmp_3/zeros:output:04lstmp_3/lstmp_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
lstmp_3/lstmp_cell_3/addAddV2%lstmp_3/lstmp_cell_3/MatMul:product:0'lstmp_3/lstmp_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+lstmp_3/lstmp_cell_3/BiasAdd/ReadVariableOpReadVariableOp4lstmp_3_lstmp_cell_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
lstmp_3/lstmp_cell_3/BiasAddBiasAddlstmp_3/lstmp_cell_3/add:z:03lstmp_3/lstmp_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$lstmp_3/lstmp_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :õ
lstmp_3/lstmp_cell_3/splitSplit-lstmp_3/lstmp_cell_3/split/split_dim:output:0%lstmp_3/lstmp_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*
	num_split~
lstmp_3/lstmp_cell_3/SigmoidSigmoid#lstmp_3/lstmp_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_3/lstmp_cell_3/Sigmoid_1Sigmoid#lstmp_3/lstmp_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_3/lstmp_cell_3/mulMul"lstmp_3/lstmp_cell_3/Sigmoid_1:y:0lstmp_3/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
lstmp_3/lstmp_cell_3/TanhTanh#lstmp_3/lstmp_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_3/lstmp_cell_3/mul_1Mul lstmp_3/lstmp_cell_3/Sigmoid:y:0lstmp_3/lstmp_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_3/lstmp_cell_3/add_1AddV2lstmp_3/lstmp_cell_3/mul:z:0lstmp_3/lstmp_cell_3/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_3/lstmp_cell_3/Sigmoid_2Sigmoid#lstmp_3/lstmp_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@u
lstmp_3/lstmp_cell_3/Tanh_1Tanhlstmp_3/lstmp_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstmp_3/lstmp_cell_3/mul_2Mul"lstmp_3/lstmp_cell_3/Sigmoid_2:y:0lstmp_3/lstmp_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
,lstmp_3/lstmp_cell_3/MatMul_2/ReadVariableOpReadVariableOp5lstmp_3_lstmp_cell_3_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype0§
lstmp_3/lstmp_cell_3/MatMul_2MatMullstmp_3/zeros:output:04lstmp_3/lstmp_cell_3/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@v
%lstmp_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   f
$lstmp_3/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstmp_3/TensorArrayV2_1TensorListReserve.lstmp_3/TensorArrayV2_1/element_shape:output:0-lstmp_3/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
lstmp_3/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstmp_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
lstmp_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : «
lstmp_3/whileWhile#lstmp_3/while/loop_counter:output:0)lstmp_3/while/maximum_iterations:output:0lstmp_3/time:output:0 lstmp_3/TensorArrayV2_1:handle:0lstmp_3/zeros:output:0lstmp_3/zeros_1:output:0 lstmp_3/strided_slice_1:output:0?lstmp_3/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstmp_3_lstmp_cell_3_matmul_readvariableop_resource5lstmp_3_lstmp_cell_3_matmul_1_readvariableop_resource4lstmp_3_lstmp_cell_3_biasadd_readvariableop_resource5lstmp_3_lstmp_cell_3_matmul_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *&
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
lstmp_3_while_body_2390*#
condR
lstmp_3_while_cond_2389*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *
parallel_iterations 
8lstmp_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   î
*lstmp_3/TensorArrayV2Stack/TensorListStackTensorListStacklstmp_3/while:output:3Alstmp_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0*
num_elementsp
lstmp_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
lstmp_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstmp_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
lstmp_3/strided_slice_3StridedSlice3lstmp_3/TensorArrayV2Stack/TensorListStack:tensor:0&lstmp_3/strided_slice_3/stack:output:0(lstmp_3/strided_slice_3/stack_1:output:0(lstmp_3/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maskm
lstmp_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ®
lstmp_3/transpose_1	Transpose3lstmp_3/TensorArrayV2Stack/TensorListStack:tensor:0!lstmp_3/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_1/MatMulMatMul lstmp_3/strided_slice_3:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp,^lstmp_2/lstmp_cell_2/BiasAdd/ReadVariableOp+^lstmp_2/lstmp_cell_2/MatMul/ReadVariableOp-^lstmp_2/lstmp_cell_2/MatMul_1/ReadVariableOp-^lstmp_2/lstmp_cell_2/MatMul_2/ReadVariableOp^lstmp_2/while,^lstmp_3/lstmp_cell_3/BiasAdd/ReadVariableOp+^lstmp_3/lstmp_cell_3/MatMul/ReadVariableOp-^lstmp_3/lstmp_cell_3/MatMul_1/ReadVariableOp-^lstmp_3/lstmp_cell_3/MatMul_2/ReadVariableOp^lstmp_3/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2Z
+lstmp_2/lstmp_cell_2/BiasAdd/ReadVariableOp+lstmp_2/lstmp_cell_2/BiasAdd/ReadVariableOp2X
*lstmp_2/lstmp_cell_2/MatMul/ReadVariableOp*lstmp_2/lstmp_cell_2/MatMul/ReadVariableOp2\
,lstmp_2/lstmp_cell_2/MatMul_1/ReadVariableOp,lstmp_2/lstmp_cell_2/MatMul_1/ReadVariableOp2\
,lstmp_2/lstmp_cell_2/MatMul_2/ReadVariableOp,lstmp_2/lstmp_cell_2/MatMul_2/ReadVariableOp2
lstmp_2/whilelstmp_2/while2Z
+lstmp_3/lstmp_cell_3/BiasAdd/ReadVariableOp+lstmp_3/lstmp_cell_3/BiasAdd/ReadVariableOp2X
*lstmp_3/lstmp_cell_3/MatMul/ReadVariableOp*lstmp_3/lstmp_cell_3/MatMul/ReadVariableOp2\
,lstmp_3/lstmp_cell_3/MatMul_1/ReadVariableOp,lstmp_3/lstmp_cell_3/MatMul_1/ReadVariableOp2\
,lstmp_3/lstmp_cell_3/MatMul_2/ReadVariableOp,lstmp_3/lstmp_cell_3/MatMul_2/ReadVariableOp2
lstmp_3/whilelstmp_3/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÂR

dlstmp_lstmp_3_while_body_1106:
6dlstmp_lstmp_3_while_dlstmp_lstmp_3_while_loop_counter@
<dlstmp_lstmp_3_while_dlstmp_lstmp_3_while_maximum_iterations$
 dlstmp_lstmp_3_while_placeholder&
"dlstmp_lstmp_3_while_placeholder_1&
"dlstmp_lstmp_3_while_placeholder_2&
"dlstmp_lstmp_3_while_placeholder_39
5dlstmp_lstmp_3_while_dlstmp_lstmp_3_strided_slice_1_0u
qdlstmp_lstmp_3_while_tensorarrayv2read_tensorlistgetitem_dlstmp_lstmp_3_tensorarrayunstack_tensorlistfromtensor_0U
Bdlstmp_lstmp_3_while_lstmp_cell_3_matmul_readvariableop_resource_0:	@W
Ddlstmp_lstmp_3_while_lstmp_cell_3_matmul_1_readvariableop_resource_0:	@R
Cdlstmp_lstmp_3_while_lstmp_cell_3_biasadd_readvariableop_resource_0:	V
Ddlstmp_lstmp_3_while_lstmp_cell_3_matmul_2_readvariableop_resource_0:@@!
dlstmp_lstmp_3_while_identity#
dlstmp_lstmp_3_while_identity_1#
dlstmp_lstmp_3_while_identity_2#
dlstmp_lstmp_3_while_identity_3#
dlstmp_lstmp_3_while_identity_4#
dlstmp_lstmp_3_while_identity_57
3dlstmp_lstmp_3_while_dlstmp_lstmp_3_strided_slice_1s
odlstmp_lstmp_3_while_tensorarrayv2read_tensorlistgetitem_dlstmp_lstmp_3_tensorarrayunstack_tensorlistfromtensorS
@dlstmp_lstmp_3_while_lstmp_cell_3_matmul_readvariableop_resource:	@U
Bdlstmp_lstmp_3_while_lstmp_cell_3_matmul_1_readvariableop_resource:	@P
Adlstmp_lstmp_3_while_lstmp_cell_3_biasadd_readvariableop_resource:	T
Bdlstmp_lstmp_3_while_lstmp_cell_3_matmul_2_readvariableop_resource:@@¢8dlstmp/lstmp_3/while/lstmp_cell_3/BiasAdd/ReadVariableOp¢7dlstmp/lstmp_3/while/lstmp_cell_3/MatMul/ReadVariableOp¢9dlstmp/lstmp_3/while/lstmp_cell_3/MatMul_1/ReadVariableOp¢9dlstmp/lstmp_3/while/lstmp_cell_3/MatMul_2/ReadVariableOp
Fdlstmp/lstmp_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ñ
8dlstmp/lstmp_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqdlstmp_lstmp_3_while_tensorarrayv2read_tensorlistgetitem_dlstmp_lstmp_3_tensorarrayunstack_tensorlistfromtensor_0 dlstmp_lstmp_3_while_placeholderOdlstmp/lstmp_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0»
7dlstmp/lstmp_3/while/lstmp_cell_3/MatMul/ReadVariableOpReadVariableOpBdlstmp_lstmp_3_while_lstmp_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	@*
dtype0ç
(dlstmp/lstmp_3/while/lstmp_cell_3/MatMulMatMul?dlstmp/lstmp_3/while/TensorArrayV2Read/TensorListGetItem:item:0?dlstmp/lstmp_3/while/lstmp_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
9dlstmp/lstmp_3/while/lstmp_cell_3/MatMul_1/ReadVariableOpReadVariableOpDdlstmp_lstmp_3_while_lstmp_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes
:	@*
dtype0Î
*dlstmp/lstmp_3/while/lstmp_cell_3/MatMul_1MatMul"dlstmp_lstmp_3_while_placeholder_2Adlstmp/lstmp_3/while/lstmp_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿË
%dlstmp/lstmp_3/while/lstmp_cell_3/addAddV22dlstmp/lstmp_3/while/lstmp_cell_3/MatMul:product:04dlstmp/lstmp_3/while/lstmp_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
8dlstmp/lstmp_3/while/lstmp_cell_3/BiasAdd/ReadVariableOpReadVariableOpCdlstmp_lstmp_3_while_lstmp_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Ô
)dlstmp/lstmp_3/while/lstmp_cell_3/BiasAddBiasAdd)dlstmp/lstmp_3/while/lstmp_cell_3/add:z:0@dlstmp/lstmp_3/while/lstmp_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
1dlstmp/lstmp_3/while/lstmp_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
'dlstmp/lstmp_3/while/lstmp_cell_3/splitSplit:dlstmp/lstmp_3/while/lstmp_cell_3/split/split_dim:output:02dlstmp/lstmp_3/while/lstmp_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*
	num_split
)dlstmp/lstmp_3/while/lstmp_cell_3/SigmoidSigmoid0dlstmp/lstmp_3/while/lstmp_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
+dlstmp/lstmp_3/while/lstmp_cell_3/Sigmoid_1Sigmoid0dlstmp/lstmp_3/while/lstmp_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@³
%dlstmp/lstmp_3/while/lstmp_cell_3/mulMul/dlstmp/lstmp_3/while/lstmp_cell_3/Sigmoid_1:y:0"dlstmp_lstmp_3_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
&dlstmp/lstmp_3/while/lstmp_cell_3/TanhTanh0dlstmp/lstmp_3/while/lstmp_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@»
'dlstmp/lstmp_3/while/lstmp_cell_3/mul_1Mul-dlstmp/lstmp_3/while/lstmp_cell_3/Sigmoid:y:0*dlstmp/lstmp_3/while/lstmp_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
'dlstmp/lstmp_3/while/lstmp_cell_3/add_1AddV2)dlstmp/lstmp_3/while/lstmp_cell_3/mul:z:0+dlstmp/lstmp_3/while/lstmp_cell_3/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
+dlstmp/lstmp_3/while/lstmp_cell_3/Sigmoid_2Sigmoid0dlstmp/lstmp_3/while/lstmp_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(dlstmp/lstmp_3/while/lstmp_cell_3/Tanh_1Tanh+dlstmp/lstmp_3/while/lstmp_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¿
'dlstmp/lstmp_3/while/lstmp_cell_3/mul_2Mul/dlstmp/lstmp_3/while/lstmp_cell_3/Sigmoid_2:y:0,dlstmp/lstmp_3/while/lstmp_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¾
9dlstmp/lstmp_3/while/lstmp_cell_3/MatMul_2/ReadVariableOpReadVariableOpDdlstmp_lstmp_3_while_lstmp_cell_3_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype0Í
*dlstmp/lstmp_3/while/lstmp_cell_3/MatMul_2MatMul"dlstmp_lstmp_3_while_placeholder_2Adlstmp/lstmp_3/while/lstmp_cell_3/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
?dlstmp/lstmp_3/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ©
9dlstmp/lstmp_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"dlstmp_lstmp_3_while_placeholder_1Hdlstmp/lstmp_3/while/TensorArrayV2Write/TensorListSetItem/index:output:0+dlstmp/lstmp_3/while/lstmp_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒ\
dlstmp/lstmp_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
dlstmp/lstmp_3/while/addAddV2 dlstmp_lstmp_3_while_placeholder#dlstmp/lstmp_3/while/add/y:output:0*
T0*
_output_shapes
: ^
dlstmp/lstmp_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :£
dlstmp/lstmp_3/while/add_1AddV26dlstmp_lstmp_3_while_dlstmp_lstmp_3_while_loop_counter%dlstmp/lstmp_3/while/add_1/y:output:0*
T0*
_output_shapes
: 
dlstmp/lstmp_3/while/IdentityIdentitydlstmp/lstmp_3/while/add_1:z:0^dlstmp/lstmp_3/while/NoOp*
T0*
_output_shapes
: ¦
dlstmp/lstmp_3/while/Identity_1Identity<dlstmp_lstmp_3_while_dlstmp_lstmp_3_while_maximum_iterations^dlstmp/lstmp_3/while/NoOp*
T0*
_output_shapes
: 
dlstmp/lstmp_3/while/Identity_2Identitydlstmp/lstmp_3/while/add:z:0^dlstmp/lstmp_3/while/NoOp*
T0*
_output_shapes
: ³
dlstmp/lstmp_3/while/Identity_3IdentityIdlstmp/lstmp_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^dlstmp/lstmp_3/while/NoOp*
T0*
_output_shapes
: ¯
dlstmp/lstmp_3/while/Identity_4Identity4dlstmp/lstmp_3/while/lstmp_cell_3/MatMul_2:product:0^dlstmp/lstmp_3/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
dlstmp/lstmp_3/while/Identity_5Identity+dlstmp/lstmp_3/while/lstmp_cell_3/add_1:z:0^dlstmp/lstmp_3/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@È
dlstmp/lstmp_3/while/NoOpNoOp9^dlstmp/lstmp_3/while/lstmp_cell_3/BiasAdd/ReadVariableOp8^dlstmp/lstmp_3/while/lstmp_cell_3/MatMul/ReadVariableOp:^dlstmp/lstmp_3/while/lstmp_cell_3/MatMul_1/ReadVariableOp:^dlstmp/lstmp_3/while/lstmp_cell_3/MatMul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "l
3dlstmp_lstmp_3_while_dlstmp_lstmp_3_strided_slice_15dlstmp_lstmp_3_while_dlstmp_lstmp_3_strided_slice_1_0"G
dlstmp_lstmp_3_while_identity&dlstmp/lstmp_3/while/Identity:output:0"K
dlstmp_lstmp_3_while_identity_1(dlstmp/lstmp_3/while/Identity_1:output:0"K
dlstmp_lstmp_3_while_identity_2(dlstmp/lstmp_3/while/Identity_2:output:0"K
dlstmp_lstmp_3_while_identity_3(dlstmp/lstmp_3/while/Identity_3:output:0"K
dlstmp_lstmp_3_while_identity_4(dlstmp/lstmp_3/while/Identity_4:output:0"K
dlstmp_lstmp_3_while_identity_5(dlstmp/lstmp_3/while/Identity_5:output:0"
Adlstmp_lstmp_3_while_lstmp_cell_3_biasadd_readvariableop_resourceCdlstmp_lstmp_3_while_lstmp_cell_3_biasadd_readvariableop_resource_0"
Bdlstmp_lstmp_3_while_lstmp_cell_3_matmul_1_readvariableop_resourceDdlstmp_lstmp_3_while_lstmp_cell_3_matmul_1_readvariableop_resource_0"
Bdlstmp_lstmp_3_while_lstmp_cell_3_matmul_2_readvariableop_resourceDdlstmp_lstmp_3_while_lstmp_cell_3_matmul_2_readvariableop_resource_0"
@dlstmp_lstmp_3_while_lstmp_cell_3_matmul_readvariableop_resourceBdlstmp_lstmp_3_while_lstmp_cell_3_matmul_readvariableop_resource_0"ä
odlstmp_lstmp_3_while_tensorarrayv2read_tensorlistgetitem_dlstmp_lstmp_3_tensorarrayunstack_tensorlistfromtensorqdlstmp_lstmp_3_while_tensorarrayv2read_tensorlistgetitem_dlstmp_lstmp_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : 2t
8dlstmp/lstmp_3/while/lstmp_cell_3/BiasAdd/ReadVariableOp8dlstmp/lstmp_3/while/lstmp_cell_3/BiasAdd/ReadVariableOp2r
7dlstmp/lstmp_3/while/lstmp_cell_3/MatMul/ReadVariableOp7dlstmp/lstmp_3/while/lstmp_cell_3/MatMul/ReadVariableOp2v
9dlstmp/lstmp_3/while/lstmp_cell_3/MatMul_1/ReadVariableOp9dlstmp/lstmp_3/while/lstmp_cell_3/MatMul_1/ReadVariableOp2v
9dlstmp/lstmp_3/while/lstmp_cell_3/MatMul_2/ReadVariableOp9dlstmp/lstmp_3/while/lstmp_cell_3/MatMul_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
¦

ù
%__inference_dlstmp_layer_call_fn_1570
input_2
unknown:	
	unknown_0:	@
	unknown_1:	
	unknown_2:@@
	unknown_3:	@
	unknown_4:	@
	unknown_5:	
	unknown_6:@@
	unknown_7:@
	unknown_8:
identity¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dlstmp_layer_call_and_return_conditional_losses_1547o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
Ô


lstmp_3_while_cond_2693,
(lstmp_3_while_lstmp_3_while_loop_counter2
.lstmp_3_while_lstmp_3_while_maximum_iterations
lstmp_3_while_placeholder
lstmp_3_while_placeholder_1
lstmp_3_while_placeholder_2
lstmp_3_while_placeholder_3.
*lstmp_3_while_less_lstmp_3_strided_slice_1B
>lstmp_3_while_lstmp_3_while_cond_2693___redundant_placeholder0B
>lstmp_3_while_lstmp_3_while_cond_2693___redundant_placeholder1B
>lstmp_3_while_lstmp_3_while_cond_2693___redundant_placeholder2B
>lstmp_3_while_lstmp_3_while_cond_2693___redundant_placeholder3B
>lstmp_3_while_lstmp_3_while_cond_2693___redundant_placeholder4
lstmp_3_while_identity

lstmp_3/while/LessLesslstmp_3_while_placeholder*lstmp_3_while_less_lstmp_3_strided_slice_1*
T0*
_output_shapes
: [
lstmp_3/while/IdentityIdentitylstmp_3/while/Less:z:0*
T0
*
_output_shapes
: "9
lstmp_3_while_identitylstmp_3/while/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*®
serving_default
?
input_24
serving_default_input_2:0ÿÿÿÿÿÿÿÿÿ;
dense_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¤
å
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
Ã
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
cell

state_spec"
_tf_keras_rnn_layer
Ã
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
cell

state_spec"
_tf_keras_rnn_layer
»
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias"
_tf_keras_layer
f
&0
'1
(2
)3
*4
+5
,6
-7
$8
%9"
trackable_list_wrapper
f
&0
'1
(2
)3
*4
+5
,6
-7
$8
%9"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
É
3trace_0
4trace_1
5trace_2
6trace_32Þ
%__inference_dlstmp_layer_call_fn_1570
%__inference_dlstmp_layer_call_fn_2156
%__inference_dlstmp_layer_call_fn_2181
%__inference_dlstmp_layer_call_fn_2038¿
¶²²
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

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z3trace_0z4trace_1z5trace_2z6trace_3
µ
7trace_0
8trace_1
9trace_2
:trace_32Ê
@__inference_dlstmp_layer_call_and_return_conditional_losses_2485
@__inference_dlstmp_layer_call_and_return_conditional_losses_2789
@__inference_dlstmp_layer_call_and_return_conditional_losses_2065
@__inference_dlstmp_layer_call_and_return_conditional_losses_2092¿
¶²²
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

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z7trace_0z8trace_1z9trace_2z:trace_3
ÊBÇ
__inference__wrapped_model_1201input_2"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
"
	optimizer
,
;serving_default"
signature_map
<
&0
'1
(2
)3"
trackable_list_wrapper
<
&0
'1
(2
)3"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

<states
=non_trainable_variables

>layers
?metrics
@layer_regularization_losses
Alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ï
Btrace_0
Ctrace_12¸
&__inference_lstmp_2_layer_call_fn_2802
&__inference_lstmp_2_layer_call_fn_2815å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zBtrace_0zCtrace_1
¥
Dtrace_0
Etrace_12î
A__inference_lstmp_2_layer_call_and_return_conditional_losses_2965
A__inference_lstmp_2_layer_call_and_return_conditional_losses_3115å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zDtrace_0zEtrace_1

F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses
L_random_generator
M
state_size

&proj_w

'kernel
(recurrent_kernel
)bias"
_tf_keras_layer
 "
trackable_list_wrapper
<
*0
+1
,2
-3"
trackable_list_wrapper
<
*0
+1
,2
-3"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

Nstates
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ï
Ttrace_0
Utrace_12¸
&__inference_lstmp_3_layer_call_fn_3130
&__inference_lstmp_3_layer_call_fn_3143å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zTtrace_0zUtrace_1
¥
Vtrace_0
Wtrace_12î
A__inference_lstmp_3_layer_call_and_return_conditional_losses_3295
A__inference_lstmp_3_layer_call_and_return_conditional_losses_3447å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zVtrace_0zWtrace_1

X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses
^_random_generator
_
state_size

*proj_w

+kernel
,recurrent_kernel
-bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
ê
etrace_02Í
&__inference_dense_1_layer_call_fn_3458¢
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
annotationsª *
 zetrace_0

ftrace_02è
A__inference_dense_1_layer_call_and_return_conditional_losses_3468¢
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
annotationsª *
 zftrace_0
 :@2dense_1/kernel
:2dense_1/bias
:@@2proj_w
.:,	2lstmp_2/lstmp_cell_2/kernel
8:6	@2%lstmp_2/lstmp_cell_2/recurrent_kernel
(:&2lstmp_2/lstmp_cell_2/bias
:@@2proj_w
.:,	@2lstmp_3/lstmp_cell_3/kernel
8:6	@2%lstmp_3/lstmp_cell_3/recurrent_kernel
(:&2lstmp_3/lstmp_cell_3/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
'
g0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
÷Bô
%__inference_dlstmp_layer_call_fn_1570input_2"¿
¶²²
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

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
öBó
%__inference_dlstmp_layer_call_fn_2156inputs"¿
¶²²
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

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
öBó
%__inference_dlstmp_layer_call_fn_2181inputs"¿
¶²²
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

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷Bô
%__inference_dlstmp_layer_call_fn_2038input_2"¿
¶²²
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

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
@__inference_dlstmp_layer_call_and_return_conditional_losses_2485inputs"¿
¶²²
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

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
@__inference_dlstmp_layer_call_and_return_conditional_losses_2789inputs"¿
¶²²
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

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
@__inference_dlstmp_layer_call_and_return_conditional_losses_2065input_2"¿
¶²²
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

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
@__inference_dlstmp_layer_call_and_return_conditional_losses_2092input_2"¿
¶²²
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

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÉBÆ
"__inference_signature_wrapper_2127input_2"
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
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
&__inference_lstmp_2_layer_call_fn_2802inputs"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
&__inference_lstmp_2_layer_call_fn_2815inputs"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¸Bµ
A__inference_lstmp_2_layer_call_and_return_conditional_losses_2965inputs"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¸Bµ
A__inference_lstmp_2_layer_call_and_return_conditional_losses_3115inputs"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
<
&0
'1
(2
)3"
trackable_list_wrapper
<
&0
'1
(2
)3"
trackable_list_wrapper
 "
trackable_list_wrapper
­
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
Ã2À½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ã2À½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
&__inference_lstmp_3_layer_call_fn_3130inputs"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
&__inference_lstmp_3_layer_call_fn_3143inputs"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¸Bµ
A__inference_lstmp_3_layer_call_and_return_conditional_losses_3295inputs"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¸Bµ
A__inference_lstmp_3_layer_call_and_return_conditional_losses_3447inputs"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
<
*0
+1
,2
-3"
trackable_list_wrapper
<
*0
+1
,2
-3"
trackable_list_wrapper
 "
trackable_list_wrapper
­
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
Ã2À½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ã2À½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
"
_generic_user_object
 "
trackable_list_wrapper
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
ÚB×
&__inference_dense_1_layer_call_fn_3458inputs"¢
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
annotationsª *
 
õBò
A__inference_dense_1_layer_call_and_return_conditional_losses_3468inputs"¢
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
annotationsª *
 
N
r	variables
s	keras_api
	ttotal
	ucount"
_tf_keras_metric
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
.
t0
u1"
trackable_list_wrapper
-
r	variables"
_generic_user_object
:  (2total
:  (2count
__inference__wrapped_model_1201u
'()&+,-*$%4¢1
*¢'
%"
input_2ÿÿÿÿÿÿÿÿÿ
ª "1ª.
,
dense_1!
dense_1ÿÿÿÿÿÿÿÿÿ¡
A__inference_dense_1_layer_call_and_return_conditional_losses_3468\$%/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 y
&__inference_dense_1_layer_call_fn_3458O$%/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿµ
@__inference_dlstmp_layer_call_and_return_conditional_losses_2065q
'()&+,-*$%<¢9
2¢/
%"
input_2ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 µ
@__inference_dlstmp_layer_call_and_return_conditional_losses_2092q
'()&+,-*$%<¢9
2¢/
%"
input_2ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ´
@__inference_dlstmp_layer_call_and_return_conditional_losses_2485p
'()&+,-*$%;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ´
@__inference_dlstmp_layer_call_and_return_conditional_losses_2789p
'()&+,-*$%;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
%__inference_dlstmp_layer_call_fn_1570d
'()&+,-*$%<¢9
2¢/
%"
input_2ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
%__inference_dlstmp_layer_call_fn_2038d
'()&+,-*$%<¢9
2¢/
%"
input_2ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
%__inference_dlstmp_layer_call_fn_2156c
'()&+,-*$%;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
%__inference_dlstmp_layer_call_fn_2181c
'()&+,-*$%;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ»
A__inference_lstmp_2_layer_call_and_return_conditional_losses_2965v'()&C¢@
9¢6
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ@
 »
A__inference_lstmp_2_layer_call_and_return_conditional_losses_3115v'()&C¢@
9¢6
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ@
 
&__inference_lstmp_2_layer_call_fn_2802i'()&C¢@
9¢6
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 

 
ª "ÿÿÿÿÿÿÿÿÿ@
&__inference_lstmp_2_layer_call_fn_2815i'()&C¢@
9¢6
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 

 
ª "ÿÿÿÿÿÿÿÿÿ@·
A__inference_lstmp_3_layer_call_and_return_conditional_losses_3295r+,-*C¢@
9¢6
$!
inputsÿÿÿÿÿÿÿÿÿ@

 
p 

 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ·
A__inference_lstmp_3_layer_call_and_return_conditional_losses_3447r+,-*C¢@
9¢6
$!
inputsÿÿÿÿÿÿÿÿÿ@

 
p

 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
&__inference_lstmp_3_layer_call_fn_3130e+,-*C¢@
9¢6
$!
inputsÿÿÿÿÿÿÿÿÿ@

 
p 

 

 
ª "ÿÿÿÿÿÿÿÿÿ@
&__inference_lstmp_3_layer_call_fn_3143e+,-*C¢@
9¢6
$!
inputsÿÿÿÿÿÿÿÿÿ@

 
p

 

 
ª "ÿÿÿÿÿÿÿÿÿ@§
"__inference_signature_wrapper_2127
'()&+,-*$%?¢<
¢ 
5ª2
0
input_2%"
input_2ÿÿÿÿÿÿÿÿÿ"1ª.
,
dense_1!
dense_1ÿÿÿÿÿÿÿÿÿ