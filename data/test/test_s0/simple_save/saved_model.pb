¶ß
Ć""
:
Add
x"T
y"T
z"T"
Ttype:
2	
A
AddV2
x"T
y"T
z"T"
Ttype:
2	
ī
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
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
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
8
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	
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
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
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
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
2
StopGradient

input"T
output"T"	
Ttype
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.15.02v1.15.0-rc3-22-g590d6eef7eßē

n
PlaceholderPlaceholder*
dtype0*'
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’
p
Placeholder_1Placeholder*
dtype0*'
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’
p
Placeholder_2Placeholder*
dtype0*'
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’
h
Placeholder_3Placeholder*
dtype0*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’
h
Placeholder_4Placeholder*
shape:’’’’’’’’’*
dtype0*#
_output_shapes
:’’’’’’’’’
Æ
5main/pi/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"     *'
_class
loc:@main/pi/dense/kernel*
dtype0*
_output_shapes
:
”
3main/pi/dense/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *Ėr÷½*'
_class
loc:@main/pi/dense/kernel
”
3main/pi/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *Ėr÷=*'
_class
loc:@main/pi/dense/kernel*
dtype0*
_output_shapes
: 
ž
=main/pi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform5main/pi/dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	*

seed *
T0*'
_class
loc:@main/pi/dense/kernel*
seed2
ī
3main/pi/dense/kernel/Initializer/random_uniform/subSub3main/pi/dense/kernel/Initializer/random_uniform/max3main/pi/dense/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
: 

3main/pi/dense/kernel/Initializer/random_uniform/mulMul=main/pi/dense/kernel/Initializer/random_uniform/RandomUniform3main/pi/dense/kernel/Initializer/random_uniform/sub*
T0*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	
ó
/main/pi/dense/kernel/Initializer/random_uniformAdd3main/pi/dense/kernel/Initializer/random_uniform/mul3main/pi/dense/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	
³
main/pi/dense/kernel
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *'
_class
loc:@main/pi/dense/kernel*
	container *
shape:	
č
main/pi/dense/kernel/AssignAssignmain/pi/dense/kernel/main/pi/dense/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*'
_class
loc:@main/pi/dense/kernel

main/pi/dense/kernel/readIdentitymain/pi/dense/kernel*
T0*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	

$main/pi/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *%
_class
loc:@main/pi/dense/bias
§
main/pi/dense/bias
VariableV2*
shared_name *%
_class
loc:@main/pi/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Ó
main/pi/dense/bias/AssignAssignmain/pi/dense/bias$main/pi/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias

main/pi/dense/bias/readIdentitymain/pi/dense/bias*
T0*%
_class
loc:@main/pi/dense/bias*
_output_shapes	
:

main/pi/dense/MatMulMatMulPlaceholdermain/pi/dense/kernel/read*
T0*(
_output_shapes
:’’’’’’’’’*
transpose_a( *
transpose_b( 

main/pi/dense/BiasAddBiasAddmain/pi/dense/MatMulmain/pi/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’
d
main/pi/dense/ReluRelumain/pi/dense/BiasAdd*
T0*(
_output_shapes
:’’’’’’’’’
³
7main/pi/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"  ,  *)
_class
loc:@main/pi/dense_1/kernel*
dtype0*
_output_shapes
:
„
5main/pi/dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *£½½*)
_class
loc:@main/pi/dense_1/kernel*
dtype0*
_output_shapes
: 
„
5main/pi/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *£½=*)
_class
loc:@main/pi/dense_1/kernel*
dtype0*
_output_shapes
: 

?main/pi/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform7main/pi/dense_1/kernel/Initializer/random_uniform/shape*
T0*)
_class
loc:@main/pi/dense_1/kernel*
seed2*
dtype0* 
_output_shapes
:
¬*

seed 
ö
5main/pi/dense_1/kernel/Initializer/random_uniform/subSub5main/pi/dense_1/kernel/Initializer/random_uniform/max5main/pi/dense_1/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@main/pi/dense_1/kernel*
_output_shapes
: 

5main/pi/dense_1/kernel/Initializer/random_uniform/mulMul?main/pi/dense_1/kernel/Initializer/random_uniform/RandomUniform5main/pi/dense_1/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@main/pi/dense_1/kernel* 
_output_shapes
:
¬
ü
1main/pi/dense_1/kernel/Initializer/random_uniformAdd5main/pi/dense_1/kernel/Initializer/random_uniform/mul5main/pi/dense_1/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@main/pi/dense_1/kernel* 
_output_shapes
:
¬
¹
main/pi/dense_1/kernel
VariableV2*
dtype0* 
_output_shapes
:
¬*
shared_name *)
_class
loc:@main/pi/dense_1/kernel*
	container *
shape:
¬
ń
main/pi/dense_1/kernel/AssignAssignmain/pi/dense_1/kernel1main/pi/dense_1/kernel/Initializer/random_uniform*
use_locking(*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
¬

main/pi/dense_1/kernel/readIdentitymain/pi/dense_1/kernel* 
_output_shapes
:
¬*
T0*)
_class
loc:@main/pi/dense_1/kernel

&main/pi/dense_1/bias/Initializer/zerosConst*
valueB¬*    *'
_class
loc:@main/pi/dense_1/bias*
dtype0*
_output_shapes	
:¬
«
main/pi/dense_1/bias
VariableV2*
	container *
shape:¬*
dtype0*
_output_shapes	
:¬*
shared_name *'
_class
loc:@main/pi/dense_1/bias
Ū
main/pi/dense_1/bias/AssignAssignmain/pi/dense_1/bias&main/pi/dense_1/bias/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
_output_shapes	
:¬

main/pi/dense_1/bias/readIdentitymain/pi/dense_1/bias*
_output_shapes	
:¬*
T0*'
_class
loc:@main/pi/dense_1/bias
Ŗ
main/pi/dense_1/MatMulMatMulmain/pi/dense/Relumain/pi/dense_1/kernel/read*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( *
T0

main/pi/dense_1/BiasAddBiasAddmain/pi/dense_1/MatMulmain/pi/dense_1/bias/read*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’¬*
T0
h
main/pi/dense_1/ReluRelumain/pi/dense_1/BiasAdd*
T0*(
_output_shapes
:’’’’’’’’’¬
³
7main/pi/dense_2/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB",     *)
_class
loc:@main/pi/dense_2/kernel
„
5main/pi/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *Ę¾*)
_class
loc:@main/pi/dense_2/kernel*
dtype0*
_output_shapes
: 
„
5main/pi/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *Ę>*)
_class
loc:@main/pi/dense_2/kernel*
dtype0*
_output_shapes
: 

?main/pi/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform7main/pi/dense_2/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	¬*

seed *
T0*)
_class
loc:@main/pi/dense_2/kernel*
seed2*
ö
5main/pi/dense_2/kernel/Initializer/random_uniform/subSub5main/pi/dense_2/kernel/Initializer/random_uniform/max5main/pi/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*)
_class
loc:@main/pi/dense_2/kernel

5main/pi/dense_2/kernel/Initializer/random_uniform/mulMul?main/pi/dense_2/kernel/Initializer/random_uniform/RandomUniform5main/pi/dense_2/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:	¬
ū
1main/pi/dense_2/kernel/Initializer/random_uniformAdd5main/pi/dense_2/kernel/Initializer/random_uniform/mul5main/pi/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
:	¬*
T0*)
_class
loc:@main/pi/dense_2/kernel
·
main/pi/dense_2/kernel
VariableV2*
dtype0*
_output_shapes
:	¬*
shared_name *)
_class
loc:@main/pi/dense_2/kernel*
	container *
shape:	¬
š
main/pi/dense_2/kernel/AssignAssignmain/pi/dense_2/kernel1main/pi/dense_2/kernel/Initializer/random_uniform*
use_locking(*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	¬

main/pi/dense_2/kernel/readIdentitymain/pi/dense_2/kernel*
T0*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:	¬

&main/pi/dense_2/bias/Initializer/zerosConst*
valueB*    *'
_class
loc:@main/pi/dense_2/bias*
dtype0*
_output_shapes
:
©
main/pi/dense_2/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *'
_class
loc:@main/pi/dense_2/bias*
	container *
shape:
Ś
main/pi/dense_2/bias/AssignAssignmain/pi/dense_2/bias&main/pi/dense_2/bias/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias

main/pi/dense_2/bias/readIdentitymain/pi/dense_2/bias*
T0*'
_class
loc:@main/pi/dense_2/bias*
_output_shapes
:
«
main/pi/dense_2/MatMulMatMulmain/pi/dense_1/Relumain/pi/dense_2/kernel/read*
transpose_b( *
T0*'
_output_shapes
:’’’’’’’’’*
transpose_a( 

main/pi/dense_2/BiasAddBiasAddmain/pi/dense_2/MatMulmain/pi/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:’’’’’’’’’
g
main/pi/dense_2/TanhTanhmain/pi/dense_2/BiasAdd*
T0*'
_output_shapes
:’’’’’’’’’
R
main/pi/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
i
main/pi/mulMulmain/pi/mul/xmain/pi/dense_2/Tanh*
T0*'
_output_shapes
:’’’’’’’’’
]
main/q/concat/axisConst*
dtype0*
_output_shapes
: *
valueB :
’’’’’’’’’

main/q/concatConcatV2PlaceholderPlaceholder_1main/q/concat/axis*
N*'
_output_shapes
:’’’’’’’’’*

Tidx0*
T0
­
4main/q/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"     *&
_class
loc:@main/q/dense/kernel*
dtype0*
_output_shapes
:

2main/q/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *Üö½*&
_class
loc:@main/q/dense/kernel*
dtype0*
_output_shapes
: 

2main/q/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *Üö=*&
_class
loc:@main/q/dense/kernel*
dtype0*
_output_shapes
: 
ū
<main/q/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform4main/q/dense/kernel/Initializer/random_uniform/shape*

seed *
T0*&
_class
loc:@main/q/dense/kernel*
seed2?*
dtype0*
_output_shapes
:	
ź
2main/q/dense/kernel/Initializer/random_uniform/subSub2main/q/dense/kernel/Initializer/random_uniform/max2main/q/dense/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@main/q/dense/kernel*
_output_shapes
: 
ż
2main/q/dense/kernel/Initializer/random_uniform/mulMul<main/q/dense/kernel/Initializer/random_uniform/RandomUniform2main/q/dense/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@main/q/dense/kernel*
_output_shapes
:	
ļ
.main/q/dense/kernel/Initializer/random_uniformAdd2main/q/dense/kernel/Initializer/random_uniform/mul2main/q/dense/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@main/q/dense/kernel*
_output_shapes
:	
±
main/q/dense/kernel
VariableV2*
shape:	*
dtype0*
_output_shapes
:	*
shared_name *&
_class
loc:@main/q/dense/kernel*
	container 
ä
main/q/dense/kernel/AssignAssignmain/q/dense/kernel.main/q/dense/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*&
_class
loc:@main/q/dense/kernel

main/q/dense/kernel/readIdentitymain/q/dense/kernel*
_output_shapes
:	*
T0*&
_class
loc:@main/q/dense/kernel

#main/q/dense/bias/Initializer/zerosConst*
valueB*    *$
_class
loc:@main/q/dense/bias*
dtype0*
_output_shapes	
:
„
main/q/dense/bias
VariableV2*
shared_name *$
_class
loc:@main/q/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Ļ
main/q/dense/bias/AssignAssignmain/q/dense/bias#main/q/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*$
_class
loc:@main/q/dense/bias

main/q/dense/bias/readIdentitymain/q/dense/bias*
_output_shapes	
:*
T0*$
_class
loc:@main/q/dense/bias

main/q/dense/MatMulMatMulmain/q/concatmain/q/dense/kernel/read*
T0*(
_output_shapes
:’’’’’’’’’*
transpose_a( *
transpose_b( 

main/q/dense/BiasAddBiasAddmain/q/dense/MatMulmain/q/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’
b
main/q/dense/ReluRelumain/q/dense/BiasAdd*
T0*(
_output_shapes
:’’’’’’’’’
±
6main/q/dense_1/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"  ,  *(
_class
loc:@main/q/dense_1/kernel
£
4main/q/dense_1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *£½½*(
_class
loc:@main/q/dense_1/kernel
£
4main/q/dense_1/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *£½=*(
_class
loc:@main/q/dense_1/kernel

>main/q/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform6main/q/dense_1/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
¬*

seed *
T0*(
_class
loc:@main/q/dense_1/kernel*
seed2P
ņ
4main/q/dense_1/kernel/Initializer/random_uniform/subSub4main/q/dense_1/kernel/Initializer/random_uniform/max4main/q/dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*(
_class
loc:@main/q/dense_1/kernel

4main/q/dense_1/kernel/Initializer/random_uniform/mulMul>main/q/dense_1/kernel/Initializer/random_uniform/RandomUniform4main/q/dense_1/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@main/q/dense_1/kernel* 
_output_shapes
:
¬
ų
0main/q/dense_1/kernel/Initializer/random_uniformAdd4main/q/dense_1/kernel/Initializer/random_uniform/mul4main/q/dense_1/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@main/q/dense_1/kernel* 
_output_shapes
:
¬
·
main/q/dense_1/kernel
VariableV2*
dtype0* 
_output_shapes
:
¬*
shared_name *(
_class
loc:@main/q/dense_1/kernel*
	container *
shape:
¬
ķ
main/q/dense_1/kernel/AssignAssignmain/q/dense_1/kernel0main/q/dense_1/kernel/Initializer/random_uniform*
T0*(
_class
loc:@main/q/dense_1/kernel*
validate_shape(* 
_output_shapes
:
¬*
use_locking(

main/q/dense_1/kernel/readIdentitymain/q/dense_1/kernel*
T0*(
_class
loc:@main/q/dense_1/kernel* 
_output_shapes
:
¬

%main/q/dense_1/bias/Initializer/zerosConst*
valueB¬*    *&
_class
loc:@main/q/dense_1/bias*
dtype0*
_output_shapes	
:¬
©
main/q/dense_1/bias
VariableV2*
dtype0*
_output_shapes	
:¬*
shared_name *&
_class
loc:@main/q/dense_1/bias*
	container *
shape:¬
×
main/q/dense_1/bias/AssignAssignmain/q/dense_1/bias%main/q/dense_1/bias/Initializer/zeros*
T0*&
_class
loc:@main/q/dense_1/bias*
validate_shape(*
_output_shapes	
:¬*
use_locking(

main/q/dense_1/bias/readIdentitymain/q/dense_1/bias*
_output_shapes	
:¬*
T0*&
_class
loc:@main/q/dense_1/bias
§
main/q/dense_1/MatMulMatMulmain/q/dense/Relumain/q/dense_1/kernel/read*
T0*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( 

main/q/dense_1/BiasAddBiasAddmain/q/dense_1/MatMulmain/q/dense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’¬
f
main/q/dense_1/ReluRelumain/q/dense_1/BiasAdd*
T0*(
_output_shapes
:’’’’’’’’’¬
±
6main/q/dense_2/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB",     *(
_class
loc:@main/q/dense_2/kernel
£
4main/q/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 * ¾*(
_class
loc:@main/q/dense_2/kernel*
dtype0*
_output_shapes
: 
£
4main/q/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 * >*(
_class
loc:@main/q/dense_2/kernel*
dtype0*
_output_shapes
: 

>main/q/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform6main/q/dense_2/kernel/Initializer/random_uniform/shape*
T0*(
_class
loc:@main/q/dense_2/kernel*
seed2a*
dtype0*
_output_shapes
:	¬*

seed 
ņ
4main/q/dense_2/kernel/Initializer/random_uniform/subSub4main/q/dense_2/kernel/Initializer/random_uniform/max4main/q/dense_2/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@main/q/dense_2/kernel*
_output_shapes
: 

4main/q/dense_2/kernel/Initializer/random_uniform/mulMul>main/q/dense_2/kernel/Initializer/random_uniform/RandomUniform4main/q/dense_2/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@main/q/dense_2/kernel*
_output_shapes
:	¬
÷
0main/q/dense_2/kernel/Initializer/random_uniformAdd4main/q/dense_2/kernel/Initializer/random_uniform/mul4main/q/dense_2/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@main/q/dense_2/kernel*
_output_shapes
:	¬
µ
main/q/dense_2/kernel
VariableV2*(
_class
loc:@main/q/dense_2/kernel*
	container *
shape:	¬*
dtype0*
_output_shapes
:	¬*
shared_name 
ģ
main/q/dense_2/kernel/AssignAssignmain/q/dense_2/kernel0main/q/dense_2/kernel/Initializer/random_uniform*
T0*(
_class
loc:@main/q/dense_2/kernel*
validate_shape(*
_output_shapes
:	¬*
use_locking(

main/q/dense_2/kernel/readIdentitymain/q/dense_2/kernel*
_output_shapes
:	¬*
T0*(
_class
loc:@main/q/dense_2/kernel

%main/q/dense_2/bias/Initializer/zerosConst*
valueB*    *&
_class
loc:@main/q/dense_2/bias*
dtype0*
_output_shapes
:
§
main/q/dense_2/bias
VariableV2*
shared_name *&
_class
loc:@main/q/dense_2/bias*
	container *
shape:*
dtype0*
_output_shapes
:
Ö
main/q/dense_2/bias/AssignAssignmain/q/dense_2/bias%main/q/dense_2/bias/Initializer/zeros*
T0*&
_class
loc:@main/q/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(

main/q/dense_2/bias/readIdentitymain/q/dense_2/bias*
_output_shapes
:*
T0*&
_class
loc:@main/q/dense_2/bias
Ø
main/q/dense_2/MatMulMatMulmain/q/dense_1/Relumain/q/dense_2/kernel/read*
transpose_b( *
T0*'
_output_shapes
:’’’’’’’’’*
transpose_a( 

main/q/dense_2/BiasAddBiasAddmain/q/dense_2/MatMulmain/q/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:’’’’’’’’’
v
main/q/SqueezeSqueezemain/q/dense_2/BiasAdd*#
_output_shapes
:’’’’’’’’’*
squeeze_dims
*
T0
_
main/q_1/concat/axisConst*
valueB :
’’’’’’’’’*
dtype0*
_output_shapes
: 

main/q_1/concatConcatV2Placeholdermain/pi/mulmain/q_1/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:’’’’’’’’’
£
main/q_1/dense/MatMulMatMulmain/q_1/concatmain/q/dense/kernel/read*
transpose_b( *
T0*(
_output_shapes
:’’’’’’’’’*
transpose_a( 

main/q_1/dense/BiasAddBiasAddmain/q_1/dense/MatMulmain/q/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’
f
main/q_1/dense/ReluRelumain/q_1/dense/BiasAdd*
T0*(
_output_shapes
:’’’’’’’’’
«
main/q_1/dense_1/MatMulMatMulmain/q_1/dense/Relumain/q/dense_1/kernel/read*
T0*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( 
 
main/q_1/dense_1/BiasAddBiasAddmain/q_1/dense_1/MatMulmain/q/dense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’¬
j
main/q_1/dense_1/ReluRelumain/q_1/dense_1/BiasAdd*(
_output_shapes
:’’’’’’’’’¬*
T0
¬
main/q_1/dense_2/MatMulMatMulmain/q_1/dense_1/Relumain/q/dense_2/kernel/read*
T0*'
_output_shapes
:’’’’’’’’’*
transpose_a( *
transpose_b( 

main/q_1/dense_2/BiasAddBiasAddmain/q_1/dense_2/MatMulmain/q/dense_2/bias/read*
data_formatNHWC*'
_output_shapes
:’’’’’’’’’*
T0
z
main/q_1/SqueezeSqueezemain/q_1/dense_2/BiasAdd*#
_output_shapes
:’’’’’’’’’*
squeeze_dims
*
T0
³
7target/pi/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"     *)
_class
loc:@target/pi/dense/kernel*
dtype0*
_output_shapes
:
„
5target/pi/dense/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *Ėr÷½*)
_class
loc:@target/pi/dense/kernel
„
5target/pi/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *Ėr÷=*)
_class
loc:@target/pi/dense/kernel*
dtype0*
_output_shapes
: 

?target/pi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform7target/pi/dense/kernel/Initializer/random_uniform/shape*
seed2}*
dtype0*
_output_shapes
:	*

seed *
T0*)
_class
loc:@target/pi/dense/kernel
ö
5target/pi/dense/kernel/Initializer/random_uniform/subSub5target/pi/dense/kernel/Initializer/random_uniform/max5target/pi/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@target/pi/dense/kernel*
_output_shapes
: 

5target/pi/dense/kernel/Initializer/random_uniform/mulMul?target/pi/dense/kernel/Initializer/random_uniform/RandomUniform5target/pi/dense/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@target/pi/dense/kernel*
_output_shapes
:	
ū
1target/pi/dense/kernel/Initializer/random_uniformAdd5target/pi/dense/kernel/Initializer/random_uniform/mul5target/pi/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@target/pi/dense/kernel*
_output_shapes
:	
·
target/pi/dense/kernel
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *)
_class
loc:@target/pi/dense/kernel*
	container *
shape:	
š
target/pi/dense/kernel/AssignAssigntarget/pi/dense/kernel1target/pi/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*)
_class
loc:@target/pi/dense/kernel*
validate_shape(*
_output_shapes
:	

target/pi/dense/kernel/readIdentitytarget/pi/dense/kernel*
T0*)
_class
loc:@target/pi/dense/kernel*
_output_shapes
:	

&target/pi/dense/bias/Initializer/zerosConst*
valueB*    *'
_class
loc:@target/pi/dense/bias*
dtype0*
_output_shapes	
:
«
target/pi/dense/bias
VariableV2*
shared_name *'
_class
loc:@target/pi/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Ū
target/pi/dense/bias/AssignAssigntarget/pi/dense/bias&target/pi/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@target/pi/dense/bias

target/pi/dense/bias/readIdentitytarget/pi/dense/bias*
T0*'
_class
loc:@target/pi/dense/bias*
_output_shapes	
:
„
target/pi/dense/MatMulMatMulPlaceholder_2target/pi/dense/kernel/read*
transpose_b( *
T0*(
_output_shapes
:’’’’’’’’’*
transpose_a( 

target/pi/dense/BiasAddBiasAddtarget/pi/dense/MatMultarget/pi/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’
h
target/pi/dense/ReluRelutarget/pi/dense/BiasAdd*
T0*(
_output_shapes
:’’’’’’’’’
·
9target/pi/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"  ,  *+
_class!
loc:@target/pi/dense_1/kernel*
dtype0*
_output_shapes
:
©
7target/pi/dense_1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *£½½*+
_class!
loc:@target/pi/dense_1/kernel
©
7target/pi/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *£½=*+
_class!
loc:@target/pi/dense_1/kernel*
dtype0*
_output_shapes
: 

Atarget/pi/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform9target/pi/dense_1/kernel/Initializer/random_uniform/shape*
T0*+
_class!
loc:@target/pi/dense_1/kernel*
seed2*
dtype0* 
_output_shapes
:
¬*

seed 
ž
7target/pi/dense_1/kernel/Initializer/random_uniform/subSub7target/pi/dense_1/kernel/Initializer/random_uniform/max7target/pi/dense_1/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@target/pi/dense_1/kernel*
_output_shapes
: 

7target/pi/dense_1/kernel/Initializer/random_uniform/mulMulAtarget/pi/dense_1/kernel/Initializer/random_uniform/RandomUniform7target/pi/dense_1/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@target/pi/dense_1/kernel* 
_output_shapes
:
¬

3target/pi/dense_1/kernel/Initializer/random_uniformAdd7target/pi/dense_1/kernel/Initializer/random_uniform/mul7target/pi/dense_1/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@target/pi/dense_1/kernel* 
_output_shapes
:
¬
½
target/pi/dense_1/kernel
VariableV2*
shared_name *+
_class!
loc:@target/pi/dense_1/kernel*
	container *
shape:
¬*
dtype0* 
_output_shapes
:
¬
ł
target/pi/dense_1/kernel/AssignAssigntarget/pi/dense_1/kernel3target/pi/dense_1/kernel/Initializer/random_uniform*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
¬

target/pi/dense_1/kernel/readIdentitytarget/pi/dense_1/kernel*
T0*+
_class!
loc:@target/pi/dense_1/kernel* 
_output_shapes
:
¬
¢
(target/pi/dense_1/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:¬*
valueB¬*    *)
_class
loc:@target/pi/dense_1/bias
Æ
target/pi/dense_1/bias
VariableV2*
shared_name *)
_class
loc:@target/pi/dense_1/bias*
	container *
shape:¬*
dtype0*
_output_shapes	
:¬
ć
target/pi/dense_1/bias/AssignAssigntarget/pi/dense_1/bias(target/pi/dense_1/bias/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@target/pi/dense_1/bias*
validate_shape(*
_output_shapes	
:¬

target/pi/dense_1/bias/readIdentitytarget/pi/dense_1/bias*
_output_shapes	
:¬*
T0*)
_class
loc:@target/pi/dense_1/bias
°
target/pi/dense_1/MatMulMatMultarget/pi/dense/Relutarget/pi/dense_1/kernel/read*
transpose_b( *
T0*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( 
„
target/pi/dense_1/BiasAddBiasAddtarget/pi/dense_1/MatMultarget/pi/dense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’¬
l
target/pi/dense_1/ReluRelutarget/pi/dense_1/BiasAdd*
T0*(
_output_shapes
:’’’’’’’’’¬
·
9target/pi/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB",     *+
_class!
loc:@target/pi/dense_2/kernel*
dtype0*
_output_shapes
:
©
7target/pi/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *Ę¾*+
_class!
loc:@target/pi/dense_2/kernel*
dtype0*
_output_shapes
: 
©
7target/pi/dense_2/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *Ę>*+
_class!
loc:@target/pi/dense_2/kernel

Atarget/pi/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform9target/pi/dense_2/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	¬*

seed *
T0*+
_class!
loc:@target/pi/dense_2/kernel*
seed2
ž
7target/pi/dense_2/kernel/Initializer/random_uniform/subSub7target/pi/dense_2/kernel/Initializer/random_uniform/max7target/pi/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*+
_class!
loc:@target/pi/dense_2/kernel

7target/pi/dense_2/kernel/Initializer/random_uniform/mulMulAtarget/pi/dense_2/kernel/Initializer/random_uniform/RandomUniform7target/pi/dense_2/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@target/pi/dense_2/kernel*
_output_shapes
:	¬

3target/pi/dense_2/kernel/Initializer/random_uniformAdd7target/pi/dense_2/kernel/Initializer/random_uniform/mul7target/pi/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
:	¬*
T0*+
_class!
loc:@target/pi/dense_2/kernel
»
target/pi/dense_2/kernel
VariableV2*
shared_name *+
_class!
loc:@target/pi/dense_2/kernel*
	container *
shape:	¬*
dtype0*
_output_shapes
:	¬
ų
target/pi/dense_2/kernel/AssignAssigntarget/pi/dense_2/kernel3target/pi/dense_2/kernel/Initializer/random_uniform*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	¬

target/pi/dense_2/kernel/readIdentitytarget/pi/dense_2/kernel*
T0*+
_class!
loc:@target/pi/dense_2/kernel*
_output_shapes
:	¬
 
(target/pi/dense_2/bias/Initializer/zerosConst*
valueB*    *)
_class
loc:@target/pi/dense_2/bias*
dtype0*
_output_shapes
:
­
target/pi/dense_2/bias
VariableV2*
shared_name *)
_class
loc:@target/pi/dense_2/bias*
	container *
shape:*
dtype0*
_output_shapes
:
ā
target/pi/dense_2/bias/AssignAssigntarget/pi/dense_2/bias(target/pi/dense_2/bias/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@target/pi/dense_2/bias*
validate_shape(*
_output_shapes
:

target/pi/dense_2/bias/readIdentitytarget/pi/dense_2/bias*
_output_shapes
:*
T0*)
_class
loc:@target/pi/dense_2/bias
±
target/pi/dense_2/MatMulMatMultarget/pi/dense_1/Relutarget/pi/dense_2/kernel/read*'
_output_shapes
:’’’’’’’’’*
transpose_a( *
transpose_b( *
T0
¤
target/pi/dense_2/BiasAddBiasAddtarget/pi/dense_2/MatMultarget/pi/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:’’’’’’’’’
k
target/pi/dense_2/TanhTanhtarget/pi/dense_2/BiasAdd*'
_output_shapes
:’’’’’’’’’*
T0
T
target/pi/mul/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
o
target/pi/mulMultarget/pi/mul/xtarget/pi/dense_2/Tanh*'
_output_shapes
:’’’’’’’’’*
T0
_
target/q/concat/axisConst*
valueB :
’’’’’’’’’*
dtype0*
_output_shapes
: 

target/q/concatConcatV2Placeholder_2Placeholder_1target/q/concat/axis*
T0*
N*'
_output_shapes
:’’’’’’’’’*

Tidx0
±
6target/q/dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"     *(
_class
loc:@target/q/dense/kernel
£
4target/q/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *Üö½*(
_class
loc:@target/q/dense/kernel*
dtype0*
_output_shapes
: 
£
4target/q/dense/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *Üö=*(
_class
loc:@target/q/dense/kernel

>target/q/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform6target/q/dense/kernel/Initializer/random_uniform/shape*
T0*(
_class
loc:@target/q/dense/kernel*
seed2“*
dtype0*
_output_shapes
:	*

seed 
ņ
4target/q/dense/kernel/Initializer/random_uniform/subSub4target/q/dense/kernel/Initializer/random_uniform/max4target/q/dense/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@target/q/dense/kernel*
_output_shapes
: 

4target/q/dense/kernel/Initializer/random_uniform/mulMul>target/q/dense/kernel/Initializer/random_uniform/RandomUniform4target/q/dense/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@target/q/dense/kernel*
_output_shapes
:	
÷
0target/q/dense/kernel/Initializer/random_uniformAdd4target/q/dense/kernel/Initializer/random_uniform/mul4target/q/dense/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@target/q/dense/kernel*
_output_shapes
:	
µ
target/q/dense/kernel
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *(
_class
loc:@target/q/dense/kernel*
	container *
shape:	
ģ
target/q/dense/kernel/AssignAssigntarget/q/dense/kernel0target/q/dense/kernel/Initializer/random_uniform*
T0*(
_class
loc:@target/q/dense/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(

target/q/dense/kernel/readIdentitytarget/q/dense/kernel*
_output_shapes
:	*
T0*(
_class
loc:@target/q/dense/kernel

%target/q/dense/bias/Initializer/zerosConst*
valueB*    *&
_class
loc:@target/q/dense/bias*
dtype0*
_output_shapes	
:
©
target/q/dense/bias
VariableV2*&
_class
loc:@target/q/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
×
target/q/dense/bias/AssignAssigntarget/q/dense/bias%target/q/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*&
_class
loc:@target/q/dense/bias

target/q/dense/bias/readIdentitytarget/q/dense/bias*
_output_shapes	
:*
T0*&
_class
loc:@target/q/dense/bias
„
target/q/dense/MatMulMatMultarget/q/concattarget/q/dense/kernel/read*
T0*(
_output_shapes
:’’’’’’’’’*
transpose_a( *
transpose_b( 

target/q/dense/BiasAddBiasAddtarget/q/dense/MatMultarget/q/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’
f
target/q/dense/ReluRelutarget/q/dense/BiasAdd*
T0*(
_output_shapes
:’’’’’’’’’
µ
8target/q/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"  ,  **
_class 
loc:@target/q/dense_1/kernel*
dtype0*
_output_shapes
:
§
6target/q/dense_1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *£½½**
_class 
loc:@target/q/dense_1/kernel
§
6target/q/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *£½=**
_class 
loc:@target/q/dense_1/kernel*
dtype0*
_output_shapes
: 

@target/q/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform8target/q/dense_1/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
¬*

seed *
T0**
_class 
loc:@target/q/dense_1/kernel*
seed2Å
ś
6target/q/dense_1/kernel/Initializer/random_uniform/subSub6target/q/dense_1/kernel/Initializer/random_uniform/max6target/q/dense_1/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@target/q/dense_1/kernel*
_output_shapes
: 

6target/q/dense_1/kernel/Initializer/random_uniform/mulMul@target/q/dense_1/kernel/Initializer/random_uniform/RandomUniform6target/q/dense_1/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
¬*
T0**
_class 
loc:@target/q/dense_1/kernel

2target/q/dense_1/kernel/Initializer/random_uniformAdd6target/q/dense_1/kernel/Initializer/random_uniform/mul6target/q/dense_1/kernel/Initializer/random_uniform/min* 
_output_shapes
:
¬*
T0**
_class 
loc:@target/q/dense_1/kernel
»
target/q/dense_1/kernel
VariableV2*
dtype0* 
_output_shapes
:
¬*
shared_name **
_class 
loc:@target/q/dense_1/kernel*
	container *
shape:
¬
õ
target/q/dense_1/kernel/AssignAssigntarget/q/dense_1/kernel2target/q/dense_1/kernel/Initializer/random_uniform*
use_locking(*
T0**
_class 
loc:@target/q/dense_1/kernel*
validate_shape(* 
_output_shapes
:
¬

target/q/dense_1/kernel/readIdentitytarget/q/dense_1/kernel* 
_output_shapes
:
¬*
T0**
_class 
loc:@target/q/dense_1/kernel
 
'target/q/dense_1/bias/Initializer/zerosConst*
valueB¬*    *(
_class
loc:@target/q/dense_1/bias*
dtype0*
_output_shapes	
:¬
­
target/q/dense_1/bias
VariableV2*
	container *
shape:¬*
dtype0*
_output_shapes	
:¬*
shared_name *(
_class
loc:@target/q/dense_1/bias
ß
target/q/dense_1/bias/AssignAssigntarget/q/dense_1/bias'target/q/dense_1/bias/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@target/q/dense_1/bias*
validate_shape(*
_output_shapes	
:¬

target/q/dense_1/bias/readIdentitytarget/q/dense_1/bias*
T0*(
_class
loc:@target/q/dense_1/bias*
_output_shapes	
:¬
­
target/q/dense_1/MatMulMatMultarget/q/dense/Relutarget/q/dense_1/kernel/read*
T0*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( 
¢
target/q/dense_1/BiasAddBiasAddtarget/q/dense_1/MatMultarget/q/dense_1/bias/read*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’¬*
T0
j
target/q/dense_1/ReluRelutarget/q/dense_1/BiasAdd*
T0*(
_output_shapes
:’’’’’’’’’¬
µ
8target/q/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB",     **
_class 
loc:@target/q/dense_2/kernel*
dtype0*
_output_shapes
:
§
6target/q/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 * ¾**
_class 
loc:@target/q/dense_2/kernel*
dtype0*
_output_shapes
: 
§
6target/q/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 * >**
_class 
loc:@target/q/dense_2/kernel*
dtype0*
_output_shapes
: 

@target/q/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform8target/q/dense_2/kernel/Initializer/random_uniform/shape*

seed *
T0**
_class 
loc:@target/q/dense_2/kernel*
seed2Ö*
dtype0*
_output_shapes
:	¬
ś
6target/q/dense_2/kernel/Initializer/random_uniform/subSub6target/q/dense_2/kernel/Initializer/random_uniform/max6target/q/dense_2/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@target/q/dense_2/kernel*
_output_shapes
: 

6target/q/dense_2/kernel/Initializer/random_uniform/mulMul@target/q/dense_2/kernel/Initializer/random_uniform/RandomUniform6target/q/dense_2/kernel/Initializer/random_uniform/sub*
_output_shapes
:	¬*
T0**
_class 
loc:@target/q/dense_2/kernel
’
2target/q/dense_2/kernel/Initializer/random_uniformAdd6target/q/dense_2/kernel/Initializer/random_uniform/mul6target/q/dense_2/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@target/q/dense_2/kernel*
_output_shapes
:	¬
¹
target/q/dense_2/kernel
VariableV2*
dtype0*
_output_shapes
:	¬*
shared_name **
_class 
loc:@target/q/dense_2/kernel*
	container *
shape:	¬
ō
target/q/dense_2/kernel/AssignAssigntarget/q/dense_2/kernel2target/q/dense_2/kernel/Initializer/random_uniform*
use_locking(*
T0**
_class 
loc:@target/q/dense_2/kernel*
validate_shape(*
_output_shapes
:	¬

target/q/dense_2/kernel/readIdentitytarget/q/dense_2/kernel*
T0**
_class 
loc:@target/q/dense_2/kernel*
_output_shapes
:	¬

'target/q/dense_2/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *(
_class
loc:@target/q/dense_2/bias
«
target/q/dense_2/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *(
_class
loc:@target/q/dense_2/bias*
	container *
shape:
Ž
target/q/dense_2/bias/AssignAssigntarget/q/dense_2/bias'target/q/dense_2/bias/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*(
_class
loc:@target/q/dense_2/bias

target/q/dense_2/bias/readIdentitytarget/q/dense_2/bias*
T0*(
_class
loc:@target/q/dense_2/bias*
_output_shapes
:
®
target/q/dense_2/MatMulMatMultarget/q/dense_1/Relutarget/q/dense_2/kernel/read*
T0*'
_output_shapes
:’’’’’’’’’*
transpose_a( *
transpose_b( 
”
target/q/dense_2/BiasAddBiasAddtarget/q/dense_2/MatMultarget/q/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:’’’’’’’’’
z
target/q/SqueezeSqueezetarget/q/dense_2/BiasAdd*
squeeze_dims
*
T0*#
_output_shapes
:’’’’’’’’’
a
target/q_1/concat/axisConst*
valueB :
’’’’’’’’’*
dtype0*
_output_shapes
: 

target/q_1/concatConcatV2Placeholder_2target/pi/multarget/q_1/concat/axis*
N*'
_output_shapes
:’’’’’’’’’*

Tidx0*
T0
©
target/q_1/dense/MatMulMatMultarget/q_1/concattarget/q/dense/kernel/read*
transpose_b( *
T0*(
_output_shapes
:’’’’’’’’’*
transpose_a( 
 
target/q_1/dense/BiasAddBiasAddtarget/q_1/dense/MatMultarget/q/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’
j
target/q_1/dense/ReluRelutarget/q_1/dense/BiasAdd*
T0*(
_output_shapes
:’’’’’’’’’
±
target/q_1/dense_1/MatMulMatMultarget/q_1/dense/Relutarget/q/dense_1/kernel/read*
T0*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b( 
¦
target/q_1/dense_1/BiasAddBiasAddtarget/q_1/dense_1/MatMultarget/q/dense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’¬
n
target/q_1/dense_1/ReluRelutarget/q_1/dense_1/BiasAdd*
T0*(
_output_shapes
:’’’’’’’’’¬
²
target/q_1/dense_2/MatMulMatMultarget/q_1/dense_1/Relutarget/q/dense_2/kernel/read*
T0*'
_output_shapes
:’’’’’’’’’*
transpose_a( *
transpose_b( 
„
target/q_1/dense_2/BiasAddBiasAddtarget/q_1/dense_2/MatMultarget/q/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:’’’’’’’’’
~
target/q_1/SqueezeSqueezetarget/q_1/dense_2/BiasAdd*
T0*#
_output_shapes
:’’’’’’’’’*
squeeze_dims

J
sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
N
subSubsub/xPlaceholder_4*
T0*#
_output_shapes
:’’’’’’’’’
J
mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *¤p}?
D
mulMulmul/xsub*
T0*#
_output_shapes
:’’’’’’’’’
S
mul_1Mulmultarget/q_1/Squeeze*
T0*#
_output_shapes
:’’’’’’’’’
P
addAddV2Placeholder_3mul_1*#
_output_shapes
:’’’’’’’’’*
T0
O
StopGradientStopGradientadd*
T0*#
_output_shapes
:’’’’’’’’’
O
ConstConst*
valueB: *
dtype0*
_output_shapes
:
c
MeanMeanmain/q_1/SqueezeConst*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
1
NegNegMean*
_output_shapes
: *
T0
X
sub_1Submain/q/SqueezeStopGradient*
T0*#
_output_shapes
:’’’’’’’’’
J
pow/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
F
powPowsub_1pow/y*
T0*#
_output_shapes
:’’’’’’’’’
Q
Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Z
Mean_1MeanpowConst_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
R
gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
X
gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
N
gradients/Neg_grad/NegNeggradients/Fill*
_output_shapes
: *
T0
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:

gradients/Mean_grad/ReshapeReshapegradients/Neg_grad/Neg!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
i
gradients/Mean_grad/ShapeShapemain/q_1/Squeeze*
T0*
out_type0*
_output_shapes
:

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*#
_output_shapes
:’’’’’’’’’*

Tmultiples0
k
gradients/Mean_grad/Shape_1Shapemain/q_1/Squeeze*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:’’’’’’’’’
}
%gradients/main/q_1/Squeeze_grad/ShapeShapemain/q_1/dense_2/BiasAdd*
_output_shapes
:*
T0*
out_type0
¶
'gradients/main/q_1/Squeeze_grad/ReshapeReshapegradients/Mean_grad/truediv%gradients/main/q_1/Squeeze_grad/Shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’
§
3gradients/main/q_1/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients/main/q_1/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes
:
 
8gradients/main/q_1/dense_2/BiasAdd_grad/tuple/group_depsNoOp(^gradients/main/q_1/Squeeze_grad/Reshape4^gradients/main/q_1/dense_2/BiasAdd_grad/BiasAddGrad

@gradients/main/q_1/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity'gradients/main/q_1/Squeeze_grad/Reshape9^gradients/main/q_1/dense_2/BiasAdd_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/q_1/Squeeze_grad/Reshape*'
_output_shapes
:’’’’’’’’’
«
Bgradients/main/q_1/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity3gradients/main/q_1/dense_2/BiasAdd_grad/BiasAddGrad9^gradients/main/q_1/dense_2/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/main/q_1/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
ī
-gradients/main/q_1/dense_2/MatMul_grad/MatMulMatMul@gradients/main/q_1/dense_2/BiasAdd_grad/tuple/control_dependencymain/q/dense_2/kernel/read*
transpose_b(*
T0*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( 
ā
/gradients/main/q_1/dense_2/MatMul_grad/MatMul_1MatMulmain/q_1/dense_1/Relu@gradients/main/q_1/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	¬*
transpose_a(
”
7gradients/main/q_1/dense_2/MatMul_grad/tuple/group_depsNoOp.^gradients/main/q_1/dense_2/MatMul_grad/MatMul0^gradients/main/q_1/dense_2/MatMul_grad/MatMul_1
©
?gradients/main/q_1/dense_2/MatMul_grad/tuple/control_dependencyIdentity-gradients/main/q_1/dense_2/MatMul_grad/MatMul8^gradients/main/q_1/dense_2/MatMul_grad/tuple/group_deps*(
_output_shapes
:’’’’’’’’’¬*
T0*@
_class6
42loc:@gradients/main/q_1/dense_2/MatMul_grad/MatMul
¦
Agradients/main/q_1/dense_2/MatMul_grad/tuple/control_dependency_1Identity/gradients/main/q_1/dense_2/MatMul_grad/MatMul_18^gradients/main/q_1/dense_2/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/main/q_1/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	¬
Ä
-gradients/main/q_1/dense_1/Relu_grad/ReluGradReluGrad?gradients/main/q_1/dense_2/MatMul_grad/tuple/control_dependencymain/q_1/dense_1/Relu*
T0*(
_output_shapes
:’’’’’’’’’¬
®
3gradients/main/q_1/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/main/q_1/dense_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:¬
¦
8gradients/main/q_1/dense_1/BiasAdd_grad/tuple/group_depsNoOp4^gradients/main/q_1/dense_1/BiasAdd_grad/BiasAddGrad.^gradients/main/q_1/dense_1/Relu_grad/ReluGrad
«
@gradients/main/q_1/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/main/q_1/dense_1/Relu_grad/ReluGrad9^gradients/main/q_1/dense_1/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/main/q_1/dense_1/Relu_grad/ReluGrad*(
_output_shapes
:’’’’’’’’’¬
¬
Bgradients/main/q_1/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity3gradients/main/q_1/dense_1/BiasAdd_grad/BiasAddGrad9^gradients/main/q_1/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:¬*
T0*F
_class<
:8loc:@gradients/main/q_1/dense_1/BiasAdd_grad/BiasAddGrad
ī
-gradients/main/q_1/dense_1/MatMul_grad/MatMulMatMul@gradients/main/q_1/dense_1/BiasAdd_grad/tuple/control_dependencymain/q/dense_1/kernel/read*(
_output_shapes
:’’’’’’’’’*
transpose_a( *
transpose_b(*
T0
į
/gradients/main/q_1/dense_1/MatMul_grad/MatMul_1MatMulmain/q_1/dense/Relu@gradients/main/q_1/dense_1/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
¬*
transpose_a(*
transpose_b( 
”
7gradients/main/q_1/dense_1/MatMul_grad/tuple/group_depsNoOp.^gradients/main/q_1/dense_1/MatMul_grad/MatMul0^gradients/main/q_1/dense_1/MatMul_grad/MatMul_1
©
?gradients/main/q_1/dense_1/MatMul_grad/tuple/control_dependencyIdentity-gradients/main/q_1/dense_1/MatMul_grad/MatMul8^gradients/main/q_1/dense_1/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/main/q_1/dense_1/MatMul_grad/MatMul*(
_output_shapes
:’’’’’’’’’
§
Agradients/main/q_1/dense_1/MatMul_grad/tuple/control_dependency_1Identity/gradients/main/q_1/dense_1/MatMul_grad/MatMul_18^gradients/main/q_1/dense_1/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/main/q_1/dense_1/MatMul_grad/MatMul_1* 
_output_shapes
:
¬
Ą
+gradients/main/q_1/dense/Relu_grad/ReluGradReluGrad?gradients/main/q_1/dense_1/MatMul_grad/tuple/control_dependencymain/q_1/dense/Relu*
T0*(
_output_shapes
:’’’’’’’’’
Ŗ
1gradients/main/q_1/dense/BiasAdd_grad/BiasAddGradBiasAddGrad+gradients/main/q_1/dense/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:
 
6gradients/main/q_1/dense/BiasAdd_grad/tuple/group_depsNoOp2^gradients/main/q_1/dense/BiasAdd_grad/BiasAddGrad,^gradients/main/q_1/dense/Relu_grad/ReluGrad
£
>gradients/main/q_1/dense/BiasAdd_grad/tuple/control_dependencyIdentity+gradients/main/q_1/dense/Relu_grad/ReluGrad7^gradients/main/q_1/dense/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:’’’’’’’’’*
T0*>
_class4
20loc:@gradients/main/q_1/dense/Relu_grad/ReluGrad
¤
@gradients/main/q_1/dense/BiasAdd_grad/tuple/control_dependency_1Identity1gradients/main/q_1/dense/BiasAdd_grad/BiasAddGrad7^gradients/main/q_1/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*D
_class:
86loc:@gradients/main/q_1/dense/BiasAdd_grad/BiasAddGrad
ē
+gradients/main/q_1/dense/MatMul_grad/MatMulMatMul>gradients/main/q_1/dense/BiasAdd_grad/tuple/control_dependencymain/q/dense/kernel/read*
transpose_b(*
T0*'
_output_shapes
:’’’’’’’’’*
transpose_a( 
Ų
-gradients/main/q_1/dense/MatMul_grad/MatMul_1MatMulmain/q_1/concat>gradients/main/q_1/dense/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 

5gradients/main/q_1/dense/MatMul_grad/tuple/group_depsNoOp,^gradients/main/q_1/dense/MatMul_grad/MatMul.^gradients/main/q_1/dense/MatMul_grad/MatMul_1
 
=gradients/main/q_1/dense/MatMul_grad/tuple/control_dependencyIdentity+gradients/main/q_1/dense/MatMul_grad/MatMul6^gradients/main/q_1/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:’’’’’’’’’*
T0*>
_class4
20loc:@gradients/main/q_1/dense/MatMul_grad/MatMul

?gradients/main/q_1/dense/MatMul_grad/tuple/control_dependency_1Identity-gradients/main/q_1/dense/MatMul_grad/MatMul_16^gradients/main/q_1/dense/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/main/q_1/dense/MatMul_grad/MatMul_1*
_output_shapes
:	
e
#gradients/main/q_1/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 

"gradients/main/q_1/concat_grad/modFloorModmain/q_1/concat/axis#gradients/main/q_1/concat_grad/Rank*
T0*
_output_shapes
: 
o
$gradients/main/q_1/concat_grad/ShapeShapePlaceholder*
T0*
out_type0*
_output_shapes
:

%gradients/main/q_1/concat_grad/ShapeNShapeNPlaceholdermain/pi/mul*
T0*
out_type0*
N* 
_output_shapes
::
Ś
+gradients/main/q_1/concat_grad/ConcatOffsetConcatOffset"gradients/main/q_1/concat_grad/mod%gradients/main/q_1/concat_grad/ShapeN'gradients/main/q_1/concat_grad/ShapeN:1*
N* 
_output_shapes
::
’
$gradients/main/q_1/concat_grad/SliceSlice=gradients/main/q_1/dense/MatMul_grad/tuple/control_dependency+gradients/main/q_1/concat_grad/ConcatOffset%gradients/main/q_1/concat_grad/ShapeN*'
_output_shapes
:’’’’’’’’’*
Index0*
T0

&gradients/main/q_1/concat_grad/Slice_1Slice=gradients/main/q_1/dense/MatMul_grad/tuple/control_dependency-gradients/main/q_1/concat_grad/ConcatOffset:1'gradients/main/q_1/concat_grad/ShapeN:1*
Index0*
T0*'
_output_shapes
:’’’’’’’’’

/gradients/main/q_1/concat_grad/tuple/group_depsNoOp%^gradients/main/q_1/concat_grad/Slice'^gradients/main/q_1/concat_grad/Slice_1

7gradients/main/q_1/concat_grad/tuple/control_dependencyIdentity$gradients/main/q_1/concat_grad/Slice0^gradients/main/q_1/concat_grad/tuple/group_deps*'
_output_shapes
:’’’’’’’’’*
T0*7
_class-
+)loc:@gradients/main/q_1/concat_grad/Slice

9gradients/main/q_1/concat_grad/tuple/control_dependency_1Identity&gradients/main/q_1/concat_grad/Slice_10^gradients/main/q_1/concat_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/main/q_1/concat_grad/Slice_1*'
_output_shapes
:’’’’’’’’’
k
 gradients/main/pi/mul_grad/ShapeShapemain/pi/mul/x*
T0*
out_type0*
_output_shapes
: 
v
"gradients/main/pi/mul_grad/Shape_1Shapemain/pi/dense_2/Tanh*
T0*
out_type0*
_output_shapes
:
Ģ
0gradients/main/pi/mul_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/main/pi/mul_grad/Shape"gradients/main/pi/mul_grad/Shape_1*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’*
T0
Ø
gradients/main/pi/mul_grad/MulMul9gradients/main/q_1/concat_grad/tuple/control_dependency_1main/pi/dense_2/Tanh*'
_output_shapes
:’’’’’’’’’*
T0
·
gradients/main/pi/mul_grad/SumSumgradients/main/pi/mul_grad/Mul0gradients/main/pi/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

"gradients/main/pi/mul_grad/ReshapeReshapegradients/main/pi/mul_grad/Sum gradients/main/pi/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
£
 gradients/main/pi/mul_grad/Mul_1Mulmain/pi/mul/x9gradients/main/q_1/concat_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:’’’’’’’’’
½
 gradients/main/pi/mul_grad/Sum_1Sum gradients/main/pi/mul_grad/Mul_12gradients/main/pi/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
µ
$gradients/main/pi/mul_grad/Reshape_1Reshape gradients/main/pi/mul_grad/Sum_1"gradients/main/pi/mul_grad/Shape_1*'
_output_shapes
:’’’’’’’’’*
T0*
Tshape0

+gradients/main/pi/mul_grad/tuple/group_depsNoOp#^gradients/main/pi/mul_grad/Reshape%^gradients/main/pi/mul_grad/Reshape_1
é
3gradients/main/pi/mul_grad/tuple/control_dependencyIdentity"gradients/main/pi/mul_grad/Reshape,^gradients/main/pi/mul_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/main/pi/mul_grad/Reshape*
_output_shapes
: 

5gradients/main/pi/mul_grad/tuple/control_dependency_1Identity$gradients/main/pi/mul_grad/Reshape_1,^gradients/main/pi/mul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/main/pi/mul_grad/Reshape_1*'
_output_shapes
:’’’’’’’’’
·
,gradients/main/pi/dense_2/Tanh_grad/TanhGradTanhGradmain/pi/dense_2/Tanh5gradients/main/pi/mul_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:’’’’’’’’’
«
2gradients/main/pi/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients/main/pi/dense_2/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes
:
£
7gradients/main/pi/dense_2/BiasAdd_grad/tuple/group_depsNoOp3^gradients/main/pi/dense_2/BiasAdd_grad/BiasAddGrad-^gradients/main/pi/dense_2/Tanh_grad/TanhGrad
¦
?gradients/main/pi/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity,gradients/main/pi/dense_2/Tanh_grad/TanhGrad8^gradients/main/pi/dense_2/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/main/pi/dense_2/Tanh_grad/TanhGrad*'
_output_shapes
:’’’’’’’’’
§
Agradients/main/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/main/pi/dense_2/BiasAdd_grad/BiasAddGrad8^gradients/main/pi/dense_2/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/main/pi/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
ķ
,gradients/main/pi/dense_2/MatMul_grad/MatMulMatMul?gradients/main/pi/dense_2/BiasAdd_grad/tuple/control_dependencymain/pi/dense_2/kernel/read*
T0*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( *
transpose_b(
ß
.gradients/main/pi/dense_2/MatMul_grad/MatMul_1MatMulmain/pi/dense_1/Relu?gradients/main/pi/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	¬*
transpose_a(*
transpose_b( 

6gradients/main/pi/dense_2/MatMul_grad/tuple/group_depsNoOp-^gradients/main/pi/dense_2/MatMul_grad/MatMul/^gradients/main/pi/dense_2/MatMul_grad/MatMul_1
„
>gradients/main/pi/dense_2/MatMul_grad/tuple/control_dependencyIdentity,gradients/main/pi/dense_2/MatMul_grad/MatMul7^gradients/main/pi/dense_2/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/main/pi/dense_2/MatMul_grad/MatMul*(
_output_shapes
:’’’’’’’’’¬
¢
@gradients/main/pi/dense_2/MatMul_grad/tuple/control_dependency_1Identity.gradients/main/pi/dense_2/MatMul_grad/MatMul_17^gradients/main/pi/dense_2/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/main/pi/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	¬
Į
,gradients/main/pi/dense_1/Relu_grad/ReluGradReluGrad>gradients/main/pi/dense_2/MatMul_grad/tuple/control_dependencymain/pi/dense_1/Relu*
T0*(
_output_shapes
:’’’’’’’’’¬
¬
2gradients/main/pi/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients/main/pi/dense_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:¬
£
7gradients/main/pi/dense_1/BiasAdd_grad/tuple/group_depsNoOp3^gradients/main/pi/dense_1/BiasAdd_grad/BiasAddGrad-^gradients/main/pi/dense_1/Relu_grad/ReluGrad
§
?gradients/main/pi/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity,gradients/main/pi/dense_1/Relu_grad/ReluGrad8^gradients/main/pi/dense_1/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/main/pi/dense_1/Relu_grad/ReluGrad*(
_output_shapes
:’’’’’’’’’¬
Ø
Agradients/main/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/main/pi/dense_1/BiasAdd_grad/BiasAddGrad8^gradients/main/pi/dense_1/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/main/pi/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:¬
ķ
,gradients/main/pi/dense_1/MatMul_grad/MatMulMatMul?gradients/main/pi/dense_1/BiasAdd_grad/tuple/control_dependencymain/pi/dense_1/kernel/read*
T0*(
_output_shapes
:’’’’’’’’’*
transpose_a( *
transpose_b(
Ž
.gradients/main/pi/dense_1/MatMul_grad/MatMul_1MatMulmain/pi/dense/Relu?gradients/main/pi/dense_1/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
¬*
transpose_a(*
transpose_b( *
T0

6gradients/main/pi/dense_1/MatMul_grad/tuple/group_depsNoOp-^gradients/main/pi/dense_1/MatMul_grad/MatMul/^gradients/main/pi/dense_1/MatMul_grad/MatMul_1
„
>gradients/main/pi/dense_1/MatMul_grad/tuple/control_dependencyIdentity,gradients/main/pi/dense_1/MatMul_grad/MatMul7^gradients/main/pi/dense_1/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/main/pi/dense_1/MatMul_grad/MatMul*(
_output_shapes
:’’’’’’’’’
£
@gradients/main/pi/dense_1/MatMul_grad/tuple/control_dependency_1Identity.gradients/main/pi/dense_1/MatMul_grad/MatMul_17^gradients/main/pi/dense_1/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/main/pi/dense_1/MatMul_grad/MatMul_1* 
_output_shapes
:
¬
½
*gradients/main/pi/dense/Relu_grad/ReluGradReluGrad>gradients/main/pi/dense_1/MatMul_grad/tuple/control_dependencymain/pi/dense/Relu*
T0*(
_output_shapes
:’’’’’’’’’
Ø
0gradients/main/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad*gradients/main/pi/dense/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:*
T0

5gradients/main/pi/dense/BiasAdd_grad/tuple/group_depsNoOp1^gradients/main/pi/dense/BiasAdd_grad/BiasAddGrad+^gradients/main/pi/dense/Relu_grad/ReluGrad

=gradients/main/pi/dense/BiasAdd_grad/tuple/control_dependencyIdentity*gradients/main/pi/dense/Relu_grad/ReluGrad6^gradients/main/pi/dense/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/main/pi/dense/Relu_grad/ReluGrad*(
_output_shapes
:’’’’’’’’’
 
?gradients/main/pi/dense/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/main/pi/dense/BiasAdd_grad/BiasAddGrad6^gradients/main/pi/dense/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/main/pi/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
ę
*gradients/main/pi/dense/MatMul_grad/MatMulMatMul=gradients/main/pi/dense/BiasAdd_grad/tuple/control_dependencymain/pi/dense/kernel/read*
transpose_b(*
T0*'
_output_shapes
:’’’’’’’’’*
transpose_a( 
Ņ
,gradients/main/pi/dense/MatMul_grad/MatMul_1MatMulPlaceholder=gradients/main/pi/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	*
transpose_a(

4gradients/main/pi/dense/MatMul_grad/tuple/group_depsNoOp+^gradients/main/pi/dense/MatMul_grad/MatMul-^gradients/main/pi/dense/MatMul_grad/MatMul_1

<gradients/main/pi/dense/MatMul_grad/tuple/control_dependencyIdentity*gradients/main/pi/dense/MatMul_grad/MatMul5^gradients/main/pi/dense/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/main/pi/dense/MatMul_grad/MatMul*'
_output_shapes
:’’’’’’’’’

>gradients/main/pi/dense/MatMul_grad/tuple/control_dependency_1Identity,gradients/main/pi/dense/MatMul_grad/MatMul_15^gradients/main/pi/dense/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/main/pi/dense/MatMul_grad/MatMul_1*
_output_shapes
:	

beta1_power/initial_valueConst*
valueB
 *fff?*%
_class
loc:@main/pi/dense/bias*
dtype0*
_output_shapes
: 

beta1_power
VariableV2*
shared_name *%
_class
loc:@main/pi/dense/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
µ
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
: 
q
beta1_power/readIdentitybeta1_power*
T0*%
_class
loc:@main/pi/dense/bias*
_output_shapes
: 

beta2_power/initial_valueConst*
valueB
 *w¾?*%
_class
loc:@main/pi/dense/bias*
dtype0*
_output_shapes
: 

beta2_power
VariableV2*
shared_name *%
_class
loc:@main/pi/dense/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
µ
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
q
beta2_power/readIdentitybeta2_power*
_output_shapes
: *
T0*%
_class
loc:@main/pi/dense/bias
µ
;main/pi/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*'
_class
loc:@main/pi/dense/kernel*
valueB"     *
dtype0*
_output_shapes
:

1main/pi/dense/kernel/Adam/Initializer/zeros/ConstConst*'
_class
loc:@main/pi/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

+main/pi/dense/kernel/Adam/Initializer/zerosFill;main/pi/dense/kernel/Adam/Initializer/zeros/shape_as_tensor1main/pi/dense/kernel/Adam/Initializer/zeros/Const*
T0*'
_class
loc:@main/pi/dense/kernel*

index_type0*
_output_shapes
:	
ø
main/pi/dense/kernel/Adam
VariableV2*'
_class
loc:@main/pi/dense/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	*
shared_name 
ī
 main/pi/dense/kernel/Adam/AssignAssignmain/pi/dense/kernel/Adam+main/pi/dense/kernel/Adam/Initializer/zeros*
T0*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(

main/pi/dense/kernel/Adam/readIdentitymain/pi/dense/kernel/Adam*
_output_shapes
:	*
T0*'
_class
loc:@main/pi/dense/kernel
·
=main/pi/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*'
_class
loc:@main/pi/dense/kernel*
valueB"     
”
3main/pi/dense/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *'
_class
loc:@main/pi/dense/kernel*
valueB
 *    

-main/pi/dense/kernel/Adam_1/Initializer/zerosFill=main/pi/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor3main/pi/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*'
_class
loc:@main/pi/dense/kernel*

index_type0*
_output_shapes
:	
ŗ
main/pi/dense/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *'
_class
loc:@main/pi/dense/kernel*
	container *
shape:	
ō
"main/pi/dense/kernel/Adam_1/AssignAssignmain/pi/dense/kernel/Adam_1-main/pi/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
_output_shapes
:	

 main/pi/dense/kernel/Adam_1/readIdentitymain/pi/dense/kernel/Adam_1*
T0*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	

)main/pi/dense/bias/Adam/Initializer/zerosConst*%
_class
loc:@main/pi/dense/bias*
valueB*    *
dtype0*
_output_shapes	
:
¬
main/pi/dense/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *%
_class
loc:@main/pi/dense/bias*
	container *
shape:
ā
main/pi/dense/bias/Adam/AssignAssignmain/pi/dense/bias/Adam)main/pi/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes	
:

main/pi/dense/bias/Adam/readIdentitymain/pi/dense/bias/Adam*
T0*%
_class
loc:@main/pi/dense/bias*
_output_shapes	
:
”
+main/pi/dense/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:*%
_class
loc:@main/pi/dense/bias*
valueB*    
®
main/pi/dense/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *%
_class
loc:@main/pi/dense/bias*
	container *
shape:
č
 main/pi/dense/bias/Adam_1/AssignAssignmain/pi/dense/bias/Adam_1+main/pi/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes	
:

main/pi/dense/bias/Adam_1/readIdentitymain/pi/dense/bias/Adam_1*
T0*%
_class
loc:@main/pi/dense/bias*
_output_shapes	
:
¹
=main/pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@main/pi/dense_1/kernel*
valueB"  ,  *
dtype0*
_output_shapes
:
£
3main/pi/dense_1/kernel/Adam/Initializer/zeros/ConstConst*)
_class
loc:@main/pi/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

-main/pi/dense_1/kernel/Adam/Initializer/zerosFill=main/pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor3main/pi/dense_1/kernel/Adam/Initializer/zeros/Const*
T0*)
_class
loc:@main/pi/dense_1/kernel*

index_type0* 
_output_shapes
:
¬
¾
main/pi/dense_1/kernel/Adam
VariableV2*
	container *
shape:
¬*
dtype0* 
_output_shapes
:
¬*
shared_name *)
_class
loc:@main/pi/dense_1/kernel
÷
"main/pi/dense_1/kernel/Adam/AssignAssignmain/pi/dense_1/kernel/Adam-main/pi/dense_1/kernel/Adam/Initializer/zeros*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
¬*
use_locking(

 main/pi/dense_1/kernel/Adam/readIdentitymain/pi/dense_1/kernel/Adam* 
_output_shapes
:
¬*
T0*)
_class
loc:@main/pi/dense_1/kernel
»
?main/pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@main/pi/dense_1/kernel*
valueB"  ,  *
dtype0*
_output_shapes
:
„
5main/pi/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *)
_class
loc:@main/pi/dense_1/kernel*
valueB
 *    

/main/pi/dense_1/kernel/Adam_1/Initializer/zerosFill?main/pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor5main/pi/dense_1/kernel/Adam_1/Initializer/zeros/Const*
T0*)
_class
loc:@main/pi/dense_1/kernel*

index_type0* 
_output_shapes
:
¬
Ą
main/pi/dense_1/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
¬*
shared_name *)
_class
loc:@main/pi/dense_1/kernel*
	container *
shape:
¬
ż
$main/pi/dense_1/kernel/Adam_1/AssignAssignmain/pi/dense_1/kernel/Adam_1/main/pi/dense_1/kernel/Adam_1/Initializer/zeros*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
¬*
use_locking(
£
"main/pi/dense_1/kernel/Adam_1/readIdentitymain/pi/dense_1/kernel/Adam_1*
T0*)
_class
loc:@main/pi/dense_1/kernel* 
_output_shapes
:
¬
£
+main/pi/dense_1/bias/Adam/Initializer/zerosConst*'
_class
loc:@main/pi/dense_1/bias*
valueB¬*    *
dtype0*
_output_shapes	
:¬
°
main/pi/dense_1/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:¬*
shared_name *'
_class
loc:@main/pi/dense_1/bias*
	container *
shape:¬
ź
 main/pi/dense_1/bias/Adam/AssignAssignmain/pi/dense_1/bias/Adam+main/pi/dense_1/bias/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
_output_shapes	
:¬

main/pi/dense_1/bias/Adam/readIdentitymain/pi/dense_1/bias/Adam*
_output_shapes	
:¬*
T0*'
_class
loc:@main/pi/dense_1/bias
„
-main/pi/dense_1/bias/Adam_1/Initializer/zerosConst*'
_class
loc:@main/pi/dense_1/bias*
valueB¬*    *
dtype0*
_output_shapes	
:¬
²
main/pi/dense_1/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:¬*
shared_name *'
_class
loc:@main/pi/dense_1/bias*
	container *
shape:¬
š
"main/pi/dense_1/bias/Adam_1/AssignAssignmain/pi/dense_1/bias/Adam_1-main/pi/dense_1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
_output_shapes	
:¬

 main/pi/dense_1/bias/Adam_1/readIdentitymain/pi/dense_1/bias/Adam_1*
T0*'
_class
loc:@main/pi/dense_1/bias*
_output_shapes	
:¬
Æ
-main/pi/dense_2/kernel/Adam/Initializer/zerosConst*)
_class
loc:@main/pi/dense_2/kernel*
valueB	¬*    *
dtype0*
_output_shapes
:	¬
¼
main/pi/dense_2/kernel/Adam
VariableV2*
shared_name *)
_class
loc:@main/pi/dense_2/kernel*
	container *
shape:	¬*
dtype0*
_output_shapes
:	¬
ö
"main/pi/dense_2/kernel/Adam/AssignAssignmain/pi/dense_2/kernel/Adam-main/pi/dense_2/kernel/Adam/Initializer/zeros*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	¬*
use_locking(

 main/pi/dense_2/kernel/Adam/readIdentitymain/pi/dense_2/kernel/Adam*
_output_shapes
:	¬*
T0*)
_class
loc:@main/pi/dense_2/kernel
±
/main/pi/dense_2/kernel/Adam_1/Initializer/zerosConst*)
_class
loc:@main/pi/dense_2/kernel*
valueB	¬*    *
dtype0*
_output_shapes
:	¬
¾
main/pi/dense_2/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes
:	¬*
shared_name *)
_class
loc:@main/pi/dense_2/kernel*
	container *
shape:	¬
ü
$main/pi/dense_2/kernel/Adam_1/AssignAssignmain/pi/dense_2/kernel/Adam_1/main/pi/dense_2/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	¬*
use_locking(*
T0*)
_class
loc:@main/pi/dense_2/kernel
¢
"main/pi/dense_2/kernel/Adam_1/readIdentitymain/pi/dense_2/kernel/Adam_1*
T0*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:	¬
”
+main/pi/dense_2/bias/Adam/Initializer/zerosConst*'
_class
loc:@main/pi/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
®
main/pi/dense_2/bias/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *'
_class
loc:@main/pi/dense_2/bias*
	container 
é
 main/pi/dense_2/bias/Adam/AssignAssignmain/pi/dense_2/bias/Adam+main/pi/dense_2/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias

main/pi/dense_2/bias/Adam/readIdentitymain/pi/dense_2/bias/Adam*
T0*'
_class
loc:@main/pi/dense_2/bias*
_output_shapes
:
£
-main/pi/dense_2/bias/Adam_1/Initializer/zerosConst*'
_class
loc:@main/pi/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
°
main/pi/dense_2/bias/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *'
_class
loc:@main/pi/dense_2/bias*
	container 
ļ
"main/pi/dense_2/bias/Adam_1/AssignAssignmain/pi/dense_2/bias/Adam_1-main/pi/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
_output_shapes
:

 main/pi/dense_2/bias/Adam_1/readIdentitymain/pi/dense_2/bias/Adam_1*
T0*'
_class
loc:@main/pi/dense_2/bias*
_output_shapes
:
W
Adam/learning_rateConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *w¾?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *wĢ+2*
dtype0*
_output_shapes
: 

*Adam/update_main/pi/dense/kernel/ApplyAdam	ApplyAdammain/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/main/pi/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@main/pi/dense/kernel*
use_nesterov( *
_output_shapes
:	

(Adam/update_main/pi/dense/bias/ApplyAdam	ApplyAdammain/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon?gradients/main/pi/dense/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0*%
_class
loc:@main/pi/dense/bias
Ŗ
,Adam/update_main/pi/dense_1/kernel/ApplyAdam	ApplyAdammain/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon@gradients/main/pi/dense_1/MatMul_grad/tuple/control_dependency_1*
T0*)
_class
loc:@main/pi/dense_1/kernel*
use_nesterov( * 
_output_shapes
:
¬*
use_locking( 

*Adam/update_main/pi/dense_1/bias/ApplyAdam	ApplyAdammain/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/main/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@main/pi/dense_1/bias*
use_nesterov( *
_output_shapes	
:¬
©
,Adam/update_main/pi/dense_2/kernel/ApplyAdam	ApplyAdammain/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon@gradients/main/pi/dense_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@main/pi/dense_2/kernel*
use_nesterov( *
_output_shapes
:	¬

*Adam/update_main/pi/dense_2/bias/ApplyAdam	ApplyAdammain/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/main/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1*
T0*'
_class
loc:@main/pi/dense_2/bias*
use_nesterov( *
_output_shapes
:*
use_locking( 

Adam/mulMulbeta1_power/read
Adam/beta1)^Adam/update_main/pi/dense/bias/ApplyAdam+^Adam/update_main/pi/dense/kernel/ApplyAdam+^Adam/update_main/pi/dense_1/bias/ApplyAdam-^Adam/update_main/pi/dense_1/kernel/ApplyAdam+^Adam/update_main/pi/dense_2/bias/ApplyAdam-^Adam/update_main/pi/dense_2/kernel/ApplyAdam*
_output_shapes
: *
T0*%
_class
loc:@main/pi/dense/bias

Adam/AssignAssignbeta1_powerAdam/mul*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking( 


Adam/mul_1Mulbeta2_power/read
Adam/beta2)^Adam/update_main/pi/dense/bias/ApplyAdam+^Adam/update_main/pi/dense/kernel/ApplyAdam+^Adam/update_main/pi/dense_1/bias/ApplyAdam-^Adam/update_main/pi/dense_1/kernel/ApplyAdam+^Adam/update_main/pi/dense_2/bias/ApplyAdam-^Adam/update_main/pi/dense_2/kernel/ApplyAdam*
T0*%
_class
loc:@main/pi/dense/bias*
_output_shapes
: 
”
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking( 
ŗ
AdamNoOp^Adam/Assign^Adam/Assign_1)^Adam/update_main/pi/dense/bias/ApplyAdam+^Adam/update_main/pi/dense/kernel/ApplyAdam+^Adam/update_main/pi/dense_1/bias/ApplyAdam-^Adam/update_main/pi/dense_1/kernel/ApplyAdam+^Adam/update_main/pi/dense_2/bias/ApplyAdam-^Adam/update_main/pi/dense_2/kernel/ApplyAdam
T
gradients_1/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
Z
gradients_1/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
u
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
_output_shapes
: *
T0*

index_type0
o
%gradients_1/Mean_1_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:

gradients_1/Mean_1_grad/ReshapeReshapegradients_1/Fill%gradients_1/Mean_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
`
gradients_1/Mean_1_grad/ShapeShapepow*
_output_shapes
:*
T0*
out_type0
¤
gradients_1/Mean_1_grad/TileTilegradients_1/Mean_1_grad/Reshapegradients_1/Mean_1_grad/Shape*
T0*#
_output_shapes
:’’’’’’’’’*

Tmultiples0
b
gradients_1/Mean_1_grad/Shape_1Shapepow*
T0*
out_type0*
_output_shapes
:
b
gradients_1/Mean_1_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
g
gradients_1/Mean_1_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
¢
gradients_1/Mean_1_grad/ProdProdgradients_1/Mean_1_grad/Shape_1gradients_1/Mean_1_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
i
gradients_1/Mean_1_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
¦
gradients_1/Mean_1_grad/Prod_1Prodgradients_1/Mean_1_grad/Shape_2gradients_1/Mean_1_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
c
!gradients_1/Mean_1_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients_1/Mean_1_grad/MaximumMaximumgradients_1/Mean_1_grad/Prod_1!gradients_1/Mean_1_grad/Maximum/y*
T0*
_output_shapes
: 

 gradients_1/Mean_1_grad/floordivFloorDivgradients_1/Mean_1_grad/Prodgradients_1/Mean_1_grad/Maximum*
T0*
_output_shapes
: 

gradients_1/Mean_1_grad/CastCast gradients_1/Mean_1_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0

gradients_1/Mean_1_grad/truedivRealDivgradients_1/Mean_1_grad/Tilegradients_1/Mean_1_grad/Cast*#
_output_shapes
:’’’’’’’’’*
T0
_
gradients_1/pow_grad/ShapeShapesub_1*
T0*
out_type0*
_output_shapes
:
_
gradients_1/pow_grad/Shape_1Shapepow/y*
_output_shapes
: *
T0*
out_type0
ŗ
*gradients_1/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pow_grad/Shapegradients_1/pow_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
u
gradients_1/pow_grad/mulMulgradients_1/Mean_1_grad/truedivpow/y*#
_output_shapes
:’’’’’’’’’*
T0
_
gradients_1/pow_grad/sub/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
c
gradients_1/pow_grad/subSubpow/ygradients_1/pow_grad/sub/y*
T0*
_output_shapes
: 
n
gradients_1/pow_grad/PowPowsub_1gradients_1/pow_grad/sub*#
_output_shapes
:’’’’’’’’’*
T0

gradients_1/pow_grad/mul_1Mulgradients_1/pow_grad/mulgradients_1/pow_grad/Pow*
T0*#
_output_shapes
:’’’’’’’’’
§
gradients_1/pow_grad/SumSumgradients_1/pow_grad/mul_1*gradients_1/pow_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients_1/pow_grad/ReshapeReshapegradients_1/pow_grad/Sumgradients_1/pow_grad/Shape*#
_output_shapes
:’’’’’’’’’*
T0*
Tshape0
c
gradients_1/pow_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
|
gradients_1/pow_grad/GreaterGreatersub_1gradients_1/pow_grad/Greater/y*#
_output_shapes
:’’’’’’’’’*
T0
i
$gradients_1/pow_grad/ones_like/ShapeShapesub_1*
T0*
out_type0*
_output_shapes
:
i
$gradients_1/pow_grad/ones_like/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
²
gradients_1/pow_grad/ones_likeFill$gradients_1/pow_grad/ones_like/Shape$gradients_1/pow_grad/ones_like/Const*
T0*

index_type0*#
_output_shapes
:’’’’’’’’’

gradients_1/pow_grad/SelectSelectgradients_1/pow_grad/Greatersub_1gradients_1/pow_grad/ones_like*#
_output_shapes
:’’’’’’’’’*
T0
j
gradients_1/pow_grad/LogLoggradients_1/pow_grad/Select*
T0*#
_output_shapes
:’’’’’’’’’
a
gradients_1/pow_grad/zeros_like	ZerosLikesub_1*
T0*#
_output_shapes
:’’’’’’’’’
®
gradients_1/pow_grad/Select_1Selectgradients_1/pow_grad/Greatergradients_1/pow_grad/Loggradients_1/pow_grad/zeros_like*
T0*#
_output_shapes
:’’’’’’’’’
u
gradients_1/pow_grad/mul_2Mulgradients_1/Mean_1_grad/truedivpow*
T0*#
_output_shapes
:’’’’’’’’’

gradients_1/pow_grad/mul_3Mulgradients_1/pow_grad/mul_2gradients_1/pow_grad/Select_1*
T0*#
_output_shapes
:’’’’’’’’’
«
gradients_1/pow_grad/Sum_1Sumgradients_1/pow_grad/mul_3,gradients_1/pow_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients_1/pow_grad/Reshape_1Reshapegradients_1/pow_grad/Sum_1gradients_1/pow_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
m
%gradients_1/pow_grad/tuple/group_depsNoOp^gradients_1/pow_grad/Reshape^gradients_1/pow_grad/Reshape_1
Ž
-gradients_1/pow_grad/tuple/control_dependencyIdentitygradients_1/pow_grad/Reshape&^gradients_1/pow_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients_1/pow_grad/Reshape*#
_output_shapes
:’’’’’’’’’
×
/gradients_1/pow_grad/tuple/control_dependency_1Identitygradients_1/pow_grad/Reshape_1&^gradients_1/pow_grad/tuple/group_deps*
_output_shapes
: *
T0*1
_class'
%#loc:@gradients_1/pow_grad/Reshape_1
j
gradients_1/sub_1_grad/ShapeShapemain/q/Squeeze*
T0*
out_type0*
_output_shapes
:
j
gradients_1/sub_1_grad/Shape_1ShapeStopGradient*
T0*
out_type0*
_output_shapes
:
Ą
,gradients_1/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/sub_1_grad/Shapegradients_1/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
¾
gradients_1/sub_1_grad/SumSum-gradients_1/pow_grad/tuple/control_dependency,gradients_1/sub_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients_1/sub_1_grad/ReshapeReshapegradients_1/sub_1_grad/Sumgradients_1/sub_1_grad/Shape*
T0*
Tshape0*#
_output_shapes
:’’’’’’’’’
~
gradients_1/sub_1_grad/NegNeg-gradients_1/pow_grad/tuple/control_dependency*#
_output_shapes
:’’’’’’’’’*
T0
Æ
gradients_1/sub_1_grad/Sum_1Sumgradients_1/sub_1_grad/Neg.gradients_1/sub_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
„
 gradients_1/sub_1_grad/Reshape_1Reshapegradients_1/sub_1_grad/Sum_1gradients_1/sub_1_grad/Shape_1*#
_output_shapes
:’’’’’’’’’*
T0*
Tshape0
s
'gradients_1/sub_1_grad/tuple/group_depsNoOp^gradients_1/sub_1_grad/Reshape!^gradients_1/sub_1_grad/Reshape_1
ę
/gradients_1/sub_1_grad/tuple/control_dependencyIdentitygradients_1/sub_1_grad/Reshape(^gradients_1/sub_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/sub_1_grad/Reshape*#
_output_shapes
:’’’’’’’’’
ģ
1gradients_1/sub_1_grad/tuple/control_dependency_1Identity gradients_1/sub_1_grad/Reshape_1(^gradients_1/sub_1_grad/tuple/group_deps*#
_output_shapes
:’’’’’’’’’*
T0*3
_class)
'%loc:@gradients_1/sub_1_grad/Reshape_1
{
%gradients_1/main/q/Squeeze_grad/ShapeShapemain/q/dense_2/BiasAdd*
_output_shapes
:*
T0*
out_type0
Ź
'gradients_1/main/q/Squeeze_grad/ReshapeReshape/gradients_1/sub_1_grad/tuple/control_dependency%gradients_1/main/q/Squeeze_grad/Shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’
§
3gradients_1/main/q/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients_1/main/q/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes
:
 
8gradients_1/main/q/dense_2/BiasAdd_grad/tuple/group_depsNoOp(^gradients_1/main/q/Squeeze_grad/Reshape4^gradients_1/main/q/dense_2/BiasAdd_grad/BiasAddGrad

@gradients_1/main/q/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity'gradients_1/main/q/Squeeze_grad/Reshape9^gradients_1/main/q/dense_2/BiasAdd_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients_1/main/q/Squeeze_grad/Reshape*'
_output_shapes
:’’’’’’’’’
«
Bgradients_1/main/q/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity3gradients_1/main/q/dense_2/BiasAdd_grad/BiasAddGrad9^gradients_1/main/q/dense_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*F
_class<
:8loc:@gradients_1/main/q/dense_2/BiasAdd_grad/BiasAddGrad
ī
-gradients_1/main/q/dense_2/MatMul_grad/MatMulMatMul@gradients_1/main/q/dense_2/BiasAdd_grad/tuple/control_dependencymain/q/dense_2/kernel/read*
transpose_b(*
T0*(
_output_shapes
:’’’’’’’’’¬*
transpose_a( 
ą
/gradients_1/main/q/dense_2/MatMul_grad/MatMul_1MatMulmain/q/dense_1/Relu@gradients_1/main/q/dense_2/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	¬*
transpose_a(*
transpose_b( *
T0
”
7gradients_1/main/q/dense_2/MatMul_grad/tuple/group_depsNoOp.^gradients_1/main/q/dense_2/MatMul_grad/MatMul0^gradients_1/main/q/dense_2/MatMul_grad/MatMul_1
©
?gradients_1/main/q/dense_2/MatMul_grad/tuple/control_dependencyIdentity-gradients_1/main/q/dense_2/MatMul_grad/MatMul8^gradients_1/main/q/dense_2/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients_1/main/q/dense_2/MatMul_grad/MatMul*(
_output_shapes
:’’’’’’’’’¬
¦
Agradients_1/main/q/dense_2/MatMul_grad/tuple/control_dependency_1Identity/gradients_1/main/q/dense_2/MatMul_grad/MatMul_18^gradients_1/main/q/dense_2/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients_1/main/q/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	¬
Ā
-gradients_1/main/q/dense_1/Relu_grad/ReluGradReluGrad?gradients_1/main/q/dense_2/MatMul_grad/tuple/control_dependencymain/q/dense_1/Relu*
T0*(
_output_shapes
:’’’’’’’’’¬
®
3gradients_1/main/q/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients_1/main/q/dense_1/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:¬*
T0
¦
8gradients_1/main/q/dense_1/BiasAdd_grad/tuple/group_depsNoOp4^gradients_1/main/q/dense_1/BiasAdd_grad/BiasAddGrad.^gradients_1/main/q/dense_1/Relu_grad/ReluGrad
«
@gradients_1/main/q/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity-gradients_1/main/q/dense_1/Relu_grad/ReluGrad9^gradients_1/main/q/dense_1/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:’’’’’’’’’¬*
T0*@
_class6
42loc:@gradients_1/main/q/dense_1/Relu_grad/ReluGrad
¬
Bgradients_1/main/q/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity3gradients_1/main/q/dense_1/BiasAdd_grad/BiasAddGrad9^gradients_1/main/q/dense_1/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_1/main/q/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:¬
ī
-gradients_1/main/q/dense_1/MatMul_grad/MatMulMatMul@gradients_1/main/q/dense_1/BiasAdd_grad/tuple/control_dependencymain/q/dense_1/kernel/read*(
_output_shapes
:’’’’’’’’’*
transpose_a( *
transpose_b(*
T0
ß
/gradients_1/main/q/dense_1/MatMul_grad/MatMul_1MatMulmain/q/dense/Relu@gradients_1/main/q/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
¬*
transpose_a(
”
7gradients_1/main/q/dense_1/MatMul_grad/tuple/group_depsNoOp.^gradients_1/main/q/dense_1/MatMul_grad/MatMul0^gradients_1/main/q/dense_1/MatMul_grad/MatMul_1
©
?gradients_1/main/q/dense_1/MatMul_grad/tuple/control_dependencyIdentity-gradients_1/main/q/dense_1/MatMul_grad/MatMul8^gradients_1/main/q/dense_1/MatMul_grad/tuple/group_deps*(
_output_shapes
:’’’’’’’’’*
T0*@
_class6
42loc:@gradients_1/main/q/dense_1/MatMul_grad/MatMul
§
Agradients_1/main/q/dense_1/MatMul_grad/tuple/control_dependency_1Identity/gradients_1/main/q/dense_1/MatMul_grad/MatMul_18^gradients_1/main/q/dense_1/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients_1/main/q/dense_1/MatMul_grad/MatMul_1* 
_output_shapes
:
¬
¾
+gradients_1/main/q/dense/Relu_grad/ReluGradReluGrad?gradients_1/main/q/dense_1/MatMul_grad/tuple/control_dependencymain/q/dense/Relu*(
_output_shapes
:’’’’’’’’’*
T0
Ŗ
1gradients_1/main/q/dense/BiasAdd_grad/BiasAddGradBiasAddGrad+gradients_1/main/q/dense/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:
 
6gradients_1/main/q/dense/BiasAdd_grad/tuple/group_depsNoOp2^gradients_1/main/q/dense/BiasAdd_grad/BiasAddGrad,^gradients_1/main/q/dense/Relu_grad/ReluGrad
£
>gradients_1/main/q/dense/BiasAdd_grad/tuple/control_dependencyIdentity+gradients_1/main/q/dense/Relu_grad/ReluGrad7^gradients_1/main/q/dense/BiasAdd_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients_1/main/q/dense/Relu_grad/ReluGrad*(
_output_shapes
:’’’’’’’’’
¤
@gradients_1/main/q/dense/BiasAdd_grad/tuple/control_dependency_1Identity1gradients_1/main/q/dense/BiasAdd_grad/BiasAddGrad7^gradients_1/main/q/dense/BiasAdd_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients_1/main/q/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
ē
+gradients_1/main/q/dense/MatMul_grad/MatMulMatMul>gradients_1/main/q/dense/BiasAdd_grad/tuple/control_dependencymain/q/dense/kernel/read*
T0*'
_output_shapes
:’’’’’’’’’*
transpose_a( *
transpose_b(
Ö
-gradients_1/main/q/dense/MatMul_grad/MatMul_1MatMulmain/q/concat>gradients_1/main/q/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	*
transpose_a(

5gradients_1/main/q/dense/MatMul_grad/tuple/group_depsNoOp,^gradients_1/main/q/dense/MatMul_grad/MatMul.^gradients_1/main/q/dense/MatMul_grad/MatMul_1
 
=gradients_1/main/q/dense/MatMul_grad/tuple/control_dependencyIdentity+gradients_1/main/q/dense/MatMul_grad/MatMul6^gradients_1/main/q/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:’’’’’’’’’*
T0*>
_class4
20loc:@gradients_1/main/q/dense/MatMul_grad/MatMul

?gradients_1/main/q/dense/MatMul_grad/tuple/control_dependency_1Identity-gradients_1/main/q/dense/MatMul_grad/MatMul_16^gradients_1/main/q/dense/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients_1/main/q/dense/MatMul_grad/MatMul_1*
_output_shapes
:	

beta1_power_1/initial_valueConst*
valueB
 *fff?*$
_class
loc:@main/q/dense/bias*
dtype0*
_output_shapes
: 

beta1_power_1
VariableV2*
shared_name *$
_class
loc:@main/q/dense/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
ŗ
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*$
_class
loc:@main/q/dense/bias
t
beta1_power_1/readIdentitybeta1_power_1*
T0*$
_class
loc:@main/q/dense/bias*
_output_shapes
: 

beta2_power_1/initial_valueConst*
valueB
 *w¾?*$
_class
loc:@main/q/dense/bias*
dtype0*
_output_shapes
: 

beta2_power_1
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *$
_class
loc:@main/q/dense/bias
ŗ
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*$
_class
loc:@main/q/dense/bias
t
beta2_power_1/readIdentitybeta2_power_1*
T0*$
_class
loc:@main/q/dense/bias*
_output_shapes
: 
³
:main/q/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@main/q/dense/kernel*
valueB"     *
dtype0*
_output_shapes
:

0main/q/dense/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *&
_class
loc:@main/q/dense/kernel*
valueB
 *    

*main/q/dense/kernel/Adam/Initializer/zerosFill:main/q/dense/kernel/Adam/Initializer/zeros/shape_as_tensor0main/q/dense/kernel/Adam/Initializer/zeros/Const*
T0*&
_class
loc:@main/q/dense/kernel*

index_type0*
_output_shapes
:	
¶
main/q/dense/kernel/Adam
VariableV2*
shared_name *&
_class
loc:@main/q/dense/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	
ź
main/q/dense/kernel/Adam/AssignAssignmain/q/dense/kernel/Adam*main/q/dense/kernel/Adam/Initializer/zeros*
T0*&
_class
loc:@main/q/dense/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(

main/q/dense/kernel/Adam/readIdentitymain/q/dense/kernel/Adam*
T0*&
_class
loc:@main/q/dense/kernel*
_output_shapes
:	
µ
<main/q/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*&
_class
loc:@main/q/dense/kernel*
valueB"     

2main/q/dense/kernel/Adam_1/Initializer/zeros/ConstConst*&
_class
loc:@main/q/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

,main/q/dense/kernel/Adam_1/Initializer/zerosFill<main/q/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor2main/q/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*&
_class
loc:@main/q/dense/kernel*

index_type0*
_output_shapes
:	
ø
main/q/dense/kernel/Adam_1
VariableV2*&
_class
loc:@main/q/dense/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	*
shared_name 
š
!main/q/dense/kernel/Adam_1/AssignAssignmain/q/dense/kernel/Adam_1,main/q/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@main/q/dense/kernel*
validate_shape(*
_output_shapes
:	

main/q/dense/kernel/Adam_1/readIdentitymain/q/dense/kernel/Adam_1*
T0*&
_class
loc:@main/q/dense/kernel*
_output_shapes
:	

(main/q/dense/bias/Adam/Initializer/zerosConst*$
_class
loc:@main/q/dense/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ŗ
main/q/dense/bias/Adam
VariableV2*
shared_name *$
_class
loc:@main/q/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Ž
main/q/dense/bias/Adam/AssignAssignmain/q/dense/bias/Adam(main/q/dense/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*$
_class
loc:@main/q/dense/bias

main/q/dense/bias/Adam/readIdentitymain/q/dense/bias/Adam*
T0*$
_class
loc:@main/q/dense/bias*
_output_shapes	
:

*main/q/dense/bias/Adam_1/Initializer/zerosConst*$
_class
loc:@main/q/dense/bias*
valueB*    *
dtype0*
_output_shapes	
:
¬
main/q/dense/bias/Adam_1
VariableV2*$
_class
loc:@main/q/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
ä
main/q/dense/bias/Adam_1/AssignAssignmain/q/dense/bias/Adam_1*main/q/dense/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*$
_class
loc:@main/q/dense/bias

main/q/dense/bias/Adam_1/readIdentitymain/q/dense/bias/Adam_1*
T0*$
_class
loc:@main/q/dense/bias*
_output_shapes	
:
·
<main/q/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*(
_class
loc:@main/q/dense_1/kernel*
valueB"  ,  
”
2main/q/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *(
_class
loc:@main/q/dense_1/kernel*
valueB
 *    

,main/q/dense_1/kernel/Adam/Initializer/zerosFill<main/q/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor2main/q/dense_1/kernel/Adam/Initializer/zeros/Const*
T0*(
_class
loc:@main/q/dense_1/kernel*

index_type0* 
_output_shapes
:
¬
¼
main/q/dense_1/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
¬*
shared_name *(
_class
loc:@main/q/dense_1/kernel*
	container *
shape:
¬
ó
!main/q/dense_1/kernel/Adam/AssignAssignmain/q/dense_1/kernel/Adam,main/q/dense_1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@main/q/dense_1/kernel*
validate_shape(* 
_output_shapes
:
¬

main/q/dense_1/kernel/Adam/readIdentitymain/q/dense_1/kernel/Adam* 
_output_shapes
:
¬*
T0*(
_class
loc:@main/q/dense_1/kernel
¹
>main/q/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*(
_class
loc:@main/q/dense_1/kernel*
valueB"  ,  *
dtype0*
_output_shapes
:
£
4main/q/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*(
_class
loc:@main/q/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

.main/q/dense_1/kernel/Adam_1/Initializer/zerosFill>main/q/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor4main/q/dense_1/kernel/Adam_1/Initializer/zeros/Const*
T0*(
_class
loc:@main/q/dense_1/kernel*

index_type0* 
_output_shapes
:
¬
¾
main/q/dense_1/kernel/Adam_1
VariableV2*(
_class
loc:@main/q/dense_1/kernel*
	container *
shape:
¬*
dtype0* 
_output_shapes
:
¬*
shared_name 
ł
#main/q/dense_1/kernel/Adam_1/AssignAssignmain/q/dense_1/kernel/Adam_1.main/q/dense_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@main/q/dense_1/kernel*
validate_shape(* 
_output_shapes
:
¬
 
!main/q/dense_1/kernel/Adam_1/readIdentitymain/q/dense_1/kernel/Adam_1* 
_output_shapes
:
¬*
T0*(
_class
loc:@main/q/dense_1/kernel
”
*main/q/dense_1/bias/Adam/Initializer/zerosConst*&
_class
loc:@main/q/dense_1/bias*
valueB¬*    *
dtype0*
_output_shapes	
:¬
®
main/q/dense_1/bias/Adam
VariableV2*
	container *
shape:¬*
dtype0*
_output_shapes	
:¬*
shared_name *&
_class
loc:@main/q/dense_1/bias
ę
main/q/dense_1/bias/Adam/AssignAssignmain/q/dense_1/bias/Adam*main/q/dense_1/bias/Adam/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@main/q/dense_1/bias*
validate_shape(*
_output_shapes	
:¬

main/q/dense_1/bias/Adam/readIdentitymain/q/dense_1/bias/Adam*
T0*&
_class
loc:@main/q/dense_1/bias*
_output_shapes	
:¬
£
,main/q/dense_1/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:¬*&
_class
loc:@main/q/dense_1/bias*
valueB¬*    
°
main/q/dense_1/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:¬*
shared_name *&
_class
loc:@main/q/dense_1/bias*
	container *
shape:¬
ģ
!main/q/dense_1/bias/Adam_1/AssignAssignmain/q/dense_1/bias/Adam_1,main/q/dense_1/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:¬*
use_locking(*
T0*&
_class
loc:@main/q/dense_1/bias

main/q/dense_1/bias/Adam_1/readIdentitymain/q/dense_1/bias/Adam_1*
_output_shapes	
:¬*
T0*&
_class
loc:@main/q/dense_1/bias
­
,main/q/dense_2/kernel/Adam/Initializer/zerosConst*(
_class
loc:@main/q/dense_2/kernel*
valueB	¬*    *
dtype0*
_output_shapes
:	¬
ŗ
main/q/dense_2/kernel/Adam
VariableV2*
shape:	¬*
dtype0*
_output_shapes
:	¬*
shared_name *(
_class
loc:@main/q/dense_2/kernel*
	container 
ņ
!main/q/dense_2/kernel/Adam/AssignAssignmain/q/dense_2/kernel/Adam,main/q/dense_2/kernel/Adam/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@main/q/dense_2/kernel*
validate_shape(*
_output_shapes
:	¬

main/q/dense_2/kernel/Adam/readIdentitymain/q/dense_2/kernel/Adam*
T0*(
_class
loc:@main/q/dense_2/kernel*
_output_shapes
:	¬
Æ
.main/q/dense_2/kernel/Adam_1/Initializer/zerosConst*(
_class
loc:@main/q/dense_2/kernel*
valueB	¬*    *
dtype0*
_output_shapes
:	¬
¼
main/q/dense_2/kernel/Adam_1
VariableV2*
	container *
shape:	¬*
dtype0*
_output_shapes
:	¬*
shared_name *(
_class
loc:@main/q/dense_2/kernel
ų
#main/q/dense_2/kernel/Adam_1/AssignAssignmain/q/dense_2/kernel/Adam_1.main/q/dense_2/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@main/q/dense_2/kernel*
validate_shape(*
_output_shapes
:	¬

!main/q/dense_2/kernel/Adam_1/readIdentitymain/q/dense_2/kernel/Adam_1*
T0*(
_class
loc:@main/q/dense_2/kernel*
_output_shapes
:	¬

*main/q/dense_2/bias/Adam/Initializer/zerosConst*&
_class
loc:@main/q/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
¬
main/q/dense_2/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *&
_class
loc:@main/q/dense_2/bias*
	container *
shape:
å
main/q/dense_2/bias/Adam/AssignAssignmain/q/dense_2/bias/Adam*main/q/dense_2/bias/Adam/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@main/q/dense_2/bias*
validate_shape(*
_output_shapes
:

main/q/dense_2/bias/Adam/readIdentitymain/q/dense_2/bias/Adam*
T0*&
_class
loc:@main/q/dense_2/bias*
_output_shapes
:
”
,main/q/dense_2/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*&
_class
loc:@main/q/dense_2/bias*
valueB*    
®
main/q/dense_2/bias/Adam_1
VariableV2*
shared_name *&
_class
loc:@main/q/dense_2/bias*
	container *
shape:*
dtype0*
_output_shapes
:
ė
!main/q/dense_2/bias/Adam_1/AssignAssignmain/q/dense_2/bias/Adam_1,main/q/dense_2/bias/Adam_1/Initializer/zeros*
T0*&
_class
loc:@main/q/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(

main/q/dense_2/bias/Adam_1/readIdentitymain/q/dense_2/bias/Adam_1*
T0*&
_class
loc:@main/q/dense_2/bias*
_output_shapes
:
Y
Adam_1/learning_rateConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
Q
Adam_1/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
Q
Adam_1/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *w¾?
S
Adam_1/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *wĢ+2
§
+Adam_1/update_main/q/dense/kernel/ApplyAdam	ApplyAdammain/q/dense/kernelmain/q/dense/kernel/Adammain/q/dense/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon?gradients_1/main/q/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_class
loc:@main/q/dense/kernel*
use_nesterov( *
_output_shapes
:	

)Adam_1/update_main/q/dense/bias/ApplyAdam	ApplyAdammain/q/dense/biasmain/q/dense/bias/Adammain/q/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon@gradients_1/main/q/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@main/q/dense/bias*
use_nesterov( *
_output_shapes	
:
“
-Adam_1/update_main/q/dense_1/kernel/ApplyAdam	ApplyAdammain/q/dense_1/kernelmain/q/dense_1/kernel/Adammain/q/dense_1/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonAgradients_1/main/q/dense_1/MatMul_grad/tuple/control_dependency_1*
use_nesterov( * 
_output_shapes
:
¬*
use_locking( *
T0*(
_class
loc:@main/q/dense_1/kernel
¦
+Adam_1/update_main/q/dense_1/bias/ApplyAdam	ApplyAdammain/q/dense_1/biasmain/q/dense_1/bias/Adammain/q/dense_1/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonBgradients_1/main/q/dense_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_class
loc:@main/q/dense_1/bias*
use_nesterov( *
_output_shapes	
:¬
³
-Adam_1/update_main/q/dense_2/kernel/ApplyAdam	ApplyAdammain/q/dense_2/kernelmain/q/dense_2/kernel/Adammain/q/dense_2/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonAgradients_1/main/q/dense_2/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:	¬*
use_locking( *
T0*(
_class
loc:@main/q/dense_2/kernel
„
+Adam_1/update_main/q/dense_2/bias/ApplyAdam	ApplyAdammain/q/dense_2/biasmain/q/dense_2/bias/Adammain/q/dense_2/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonBgradients_1/main/q/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_class
loc:@main/q/dense_2/bias*
use_nesterov( *
_output_shapes
:


Adam_1/mulMulbeta1_power_1/readAdam_1/beta1*^Adam_1/update_main/q/dense/bias/ApplyAdam,^Adam_1/update_main/q/dense/kernel/ApplyAdam,^Adam_1/update_main/q/dense_1/bias/ApplyAdam.^Adam_1/update_main/q/dense_1/kernel/ApplyAdam,^Adam_1/update_main/q/dense_2/bias/ApplyAdam.^Adam_1/update_main/q/dense_2/kernel/ApplyAdam*
_output_shapes
: *
T0*$
_class
loc:@main/q/dense/bias
¢
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
use_locking( *
T0*$
_class
loc:@main/q/dense/bias*
validate_shape(*
_output_shapes
: 

Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2*^Adam_1/update_main/q/dense/bias/ApplyAdam,^Adam_1/update_main/q/dense/kernel/ApplyAdam,^Adam_1/update_main/q/dense_1/bias/ApplyAdam.^Adam_1/update_main/q/dense_1/kernel/ApplyAdam,^Adam_1/update_main/q/dense_2/bias/ApplyAdam.^Adam_1/update_main/q/dense_2/kernel/ApplyAdam*
_output_shapes
: *
T0*$
_class
loc:@main/q/dense/bias
¦
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
T0*$
_class
loc:@main/q/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking( 
Ę
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_1*^Adam_1/update_main/q/dense/bias/ApplyAdam,^Adam_1/update_main/q/dense/kernel/ApplyAdam,^Adam_1/update_main/q/dense_1/bias/ApplyAdam.^Adam_1/update_main/q/dense_1/kernel/ApplyAdam,^Adam_1/update_main/q/dense_2/bias/ApplyAdam.^Adam_1/update_main/q/dense_2/kernel/ApplyAdam
L
mul_2/xConst*
valueB
 *Rø~?*
dtype0*
_output_shapes
: 
\
mul_2Mulmul_2/xtarget/pi/dense/kernel/read*
T0*
_output_shapes
:	
L
mul_3/xConst*
valueB
 *
×£;*
dtype0*
_output_shapes
: 
Z
mul_3Mulmul_3/xmain/pi/dense/kernel/read*
T0*
_output_shapes
:	
F
add_1AddV2mul_2mul_3*
T0*
_output_shapes
:	
­
AssignAssigntarget/pi/dense/kerneladd_1*
use_locking(*
T0*)
_class
loc:@target/pi/dense/kernel*
validate_shape(*
_output_shapes
:	
L
mul_4/xConst*
valueB
 *Rø~?*
dtype0*
_output_shapes
: 
V
mul_4Mulmul_4/xtarget/pi/dense/bias/read*
_output_shapes	
:*
T0
L
mul_5/xConst*
valueB
 *
×£;*
dtype0*
_output_shapes
: 
T
mul_5Mulmul_5/xmain/pi/dense/bias/read*
T0*
_output_shapes	
:
B
add_2AddV2mul_4mul_5*
_output_shapes	
:*
T0
§
Assign_1Assigntarget/pi/dense/biasadd_2*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@target/pi/dense/bias
L
mul_6/xConst*
valueB
 *Rø~?*
dtype0*
_output_shapes
: 
_
mul_6Mulmul_6/xtarget/pi/dense_1/kernel/read* 
_output_shapes
:
¬*
T0
L
mul_7/xConst*
valueB
 *
×£;*
dtype0*
_output_shapes
: 
]
mul_7Mulmul_7/xmain/pi/dense_1/kernel/read*
T0* 
_output_shapes
:
¬
G
add_3AddV2mul_6mul_7*
T0* 
_output_shapes
:
¬
“
Assign_2Assigntarget/pi/dense_1/kerneladd_3*
validate_shape(* 
_output_shapes
:
¬*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_1/kernel
L
mul_8/xConst*
valueB
 *Rø~?*
dtype0*
_output_shapes
: 
X
mul_8Mulmul_8/xtarget/pi/dense_1/bias/read*
T0*
_output_shapes	
:¬
L
mul_9/xConst*
valueB
 *
×£;*
dtype0*
_output_shapes
: 
V
mul_9Mulmul_9/xmain/pi/dense_1/bias/read*
T0*
_output_shapes	
:¬
B
add_4AddV2mul_8mul_9*
T0*
_output_shapes	
:¬
«
Assign_3Assigntarget/pi/dense_1/biasadd_4*
use_locking(*
T0*)
_class
loc:@target/pi/dense_1/bias*
validate_shape(*
_output_shapes	
:¬
M
mul_10/xConst*
valueB
 *Rø~?*
dtype0*
_output_shapes
: 
`
mul_10Mulmul_10/xtarget/pi/dense_2/kernel/read*
T0*
_output_shapes
:	¬
M
mul_11/xConst*
valueB
 *
×£;*
dtype0*
_output_shapes
: 
^
mul_11Mulmul_11/xmain/pi/dense_2/kernel/read*
T0*
_output_shapes
:	¬
H
add_5AddV2mul_10mul_11*
T0*
_output_shapes
:	¬
³
Assign_4Assigntarget/pi/dense_2/kerneladd_5*
validate_shape(*
_output_shapes
:	¬*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_2/kernel
M
mul_12/xConst*
valueB
 *Rø~?*
dtype0*
_output_shapes
: 
Y
mul_12Mulmul_12/xtarget/pi/dense_2/bias/read*
T0*
_output_shapes
:
M
mul_13/xConst*
dtype0*
_output_shapes
: *
valueB
 *
×£;
W
mul_13Mulmul_13/xmain/pi/dense_2/bias/read*
T0*
_output_shapes
:
C
add_6AddV2mul_12mul_13*
T0*
_output_shapes
:
Ŗ
Assign_5Assigntarget/pi/dense_2/biasadd_6*
use_locking(*
T0*)
_class
loc:@target/pi/dense_2/bias*
validate_shape(*
_output_shapes
:
M
mul_14/xConst*
valueB
 *Rø~?*
dtype0*
_output_shapes
: 
]
mul_14Mulmul_14/xtarget/q/dense/kernel/read*
_output_shapes
:	*
T0
M
mul_15/xConst*
valueB
 *
×£;*
dtype0*
_output_shapes
: 
[
mul_15Mulmul_15/xmain/q/dense/kernel/read*
_output_shapes
:	*
T0
H
add_7AddV2mul_14mul_15*
T0*
_output_shapes
:	
­
Assign_6Assigntarget/q/dense/kerneladd_7*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*(
_class
loc:@target/q/dense/kernel
M
mul_16/xConst*
dtype0*
_output_shapes
: *
valueB
 *Rø~?
W
mul_16Mulmul_16/xtarget/q/dense/bias/read*
T0*
_output_shapes	
:
M
mul_17/xConst*
valueB
 *
×£;*
dtype0*
_output_shapes
: 
U
mul_17Mulmul_17/xmain/q/dense/bias/read*
T0*
_output_shapes	
:
D
add_8AddV2mul_16mul_17*
T0*
_output_shapes	
:
„
Assign_7Assigntarget/q/dense/biasadd_8*
T0*&
_class
loc:@target/q/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
M
mul_18/xConst*
dtype0*
_output_shapes
: *
valueB
 *Rø~?
`
mul_18Mulmul_18/xtarget/q/dense_1/kernel/read*
T0* 
_output_shapes
:
¬
M
mul_19/xConst*
valueB
 *
×£;*
dtype0*
_output_shapes
: 
^
mul_19Mulmul_19/xmain/q/dense_1/kernel/read*
T0* 
_output_shapes
:
¬
I
add_9AddV2mul_18mul_19*
T0* 
_output_shapes
:
¬
²
Assign_8Assigntarget/q/dense_1/kerneladd_9*
validate_shape(* 
_output_shapes
:
¬*
use_locking(*
T0**
_class 
loc:@target/q/dense_1/kernel
M
mul_20/xConst*
dtype0*
_output_shapes
: *
valueB
 *Rø~?
Y
mul_20Mulmul_20/xtarget/q/dense_1/bias/read*
T0*
_output_shapes	
:¬
M
mul_21/xConst*
valueB
 *
×£;*
dtype0*
_output_shapes
: 
W
mul_21Mulmul_21/xmain/q/dense_1/bias/read*
T0*
_output_shapes	
:¬
E
add_10AddV2mul_20mul_21*
T0*
_output_shapes	
:¬
Ŗ
Assign_9Assigntarget/q/dense_1/biasadd_10*
use_locking(*
T0*(
_class
loc:@target/q/dense_1/bias*
validate_shape(*
_output_shapes	
:¬
M
mul_22/xConst*
valueB
 *Rø~?*
dtype0*
_output_shapes
: 
_
mul_22Mulmul_22/xtarget/q/dense_2/kernel/read*
T0*
_output_shapes
:	¬
M
mul_23/xConst*
dtype0*
_output_shapes
: *
valueB
 *
×£;
]
mul_23Mulmul_23/xmain/q/dense_2/kernel/read*
T0*
_output_shapes
:	¬
I
add_11AddV2mul_22mul_23*
_output_shapes
:	¬*
T0
³
	Assign_10Assigntarget/q/dense_2/kerneladd_11*
use_locking(*
T0**
_class 
loc:@target/q/dense_2/kernel*
validate_shape(*
_output_shapes
:	¬
M
mul_24/xConst*
valueB
 *Rø~?*
dtype0*
_output_shapes
: 
X
mul_24Mulmul_24/xtarget/q/dense_2/bias/read*
T0*
_output_shapes
:
M
mul_25/xConst*
valueB
 *
×£;*
dtype0*
_output_shapes
: 
V
mul_25Mulmul_25/xmain/q/dense_2/bias/read*
T0*
_output_shapes
:
D
add_12AddV2mul_24mul_25*
_output_shapes
:*
T0
Ŗ
	Assign_11Assigntarget/q/dense_2/biasadd_12*
use_locking(*
T0*(
_class
loc:@target/q/dense_2/bias*
validate_shape(*
_output_shapes
:


group_depsNoOp^Assign	^Assign_1
^Assign_10
^Assign_11	^Assign_2	^Assign_3	^Assign_4	^Assign_5	^Assign_6	^Assign_7	^Assign_8	^Assign_9
Ä
	Assign_12Assigntarget/pi/dense/kernelmain/pi/dense/kernel/read*
use_locking(*
T0*)
_class
loc:@target/pi/dense/kernel*
validate_shape(*
_output_shapes
:	
ŗ
	Assign_13Assigntarget/pi/dense/biasmain/pi/dense/bias/read*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@target/pi/dense/bias
Ė
	Assign_14Assigntarget/pi/dense_1/kernelmain/pi/dense_1/kernel/read*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
¬
Ą
	Assign_15Assigntarget/pi/dense_1/biasmain/pi/dense_1/bias/read*
use_locking(*
T0*)
_class
loc:@target/pi/dense_1/bias*
validate_shape(*
_output_shapes	
:¬
Ź
	Assign_16Assigntarget/pi/dense_2/kernelmain/pi/dense_2/kernel/read*
validate_shape(*
_output_shapes
:	¬*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_2/kernel
æ
	Assign_17Assigntarget/pi/dense_2/biasmain/pi/dense_2/bias/read*
use_locking(*
T0*)
_class
loc:@target/pi/dense_2/bias*
validate_shape(*
_output_shapes
:
Į
	Assign_18Assigntarget/q/dense/kernelmain/q/dense/kernel/read*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*(
_class
loc:@target/q/dense/kernel
·
	Assign_19Assigntarget/q/dense/biasmain/q/dense/bias/read*
T0*&
_class
loc:@target/q/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Č
	Assign_20Assigntarget/q/dense_1/kernelmain/q/dense_1/kernel/read*
use_locking(*
T0**
_class 
loc:@target/q/dense_1/kernel*
validate_shape(* 
_output_shapes
:
¬
½
	Assign_21Assigntarget/q/dense_1/biasmain/q/dense_1/bias/read*
T0*(
_class
loc:@target/q/dense_1/bias*
validate_shape(*
_output_shapes	
:¬*
use_locking(
Ē
	Assign_22Assigntarget/q/dense_2/kernelmain/q/dense_2/kernel/read*
use_locking(*
T0**
_class 
loc:@target/q/dense_2/kernel*
validate_shape(*
_output_shapes
:	¬
¼
	Assign_23Assigntarget/q/dense_2/biasmain/q/dense_2/bias/read*
T0*(
_class
loc:@target/q/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
¤
group_deps_1NoOp
^Assign_12
^Assign_13
^Assign_14
^Assign_15
^Assign_16
^Assign_17
^Assign_18
^Assign_19
^Assign_20
^Assign_21
^Assign_22
^Assign_23
¤
initNoOp^beta1_power/Assign^beta1_power_1/Assign^beta2_power/Assign^beta2_power_1/Assign^main/pi/dense/bias/Adam/Assign!^main/pi/dense/bias/Adam_1/Assign^main/pi/dense/bias/Assign!^main/pi/dense/kernel/Adam/Assign#^main/pi/dense/kernel/Adam_1/Assign^main/pi/dense/kernel/Assign!^main/pi/dense_1/bias/Adam/Assign#^main/pi/dense_1/bias/Adam_1/Assign^main/pi/dense_1/bias/Assign#^main/pi/dense_1/kernel/Adam/Assign%^main/pi/dense_1/kernel/Adam_1/Assign^main/pi/dense_1/kernel/Assign!^main/pi/dense_2/bias/Adam/Assign#^main/pi/dense_2/bias/Adam_1/Assign^main/pi/dense_2/bias/Assign#^main/pi/dense_2/kernel/Adam/Assign%^main/pi/dense_2/kernel/Adam_1/Assign^main/pi/dense_2/kernel/Assign^main/q/dense/bias/Adam/Assign ^main/q/dense/bias/Adam_1/Assign^main/q/dense/bias/Assign ^main/q/dense/kernel/Adam/Assign"^main/q/dense/kernel/Adam_1/Assign^main/q/dense/kernel/Assign ^main/q/dense_1/bias/Adam/Assign"^main/q/dense_1/bias/Adam_1/Assign^main/q/dense_1/bias/Assign"^main/q/dense_1/kernel/Adam/Assign$^main/q/dense_1/kernel/Adam_1/Assign^main/q/dense_1/kernel/Assign ^main/q/dense_2/bias/Adam/Assign"^main/q/dense_2/bias/Adam_1/Assign^main/q/dense_2/bias/Assign"^main/q/dense_2/kernel/Adam/Assign$^main/q/dense_2/kernel/Adam_1/Assign^main/q/dense_2/kernel/Assign^target/pi/dense/bias/Assign^target/pi/dense/kernel/Assign^target/pi/dense_1/bias/Assign ^target/pi/dense_1/kernel/Assign^target/pi/dense_2/bias/Assign ^target/pi/dense_2/kernel/Assign^target/q/dense/bias/Assign^target/q/dense/kernel/Assign^target/q/dense_1/bias/Assign^target/q/dense_1/kernel/Assign^target/q/dense_2/bias/Assign^target/q/dense_2/kernel/Assign
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_04fbefd4784d4983bdda92d00af1cc02/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
Ś

save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:4*

value
B
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel
Ė
save/SaveV2/shape_and_slicesConst*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:4

save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1main/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1main/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1main/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1main/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1main/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1main/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1main/q/dense/biasmain/q/dense/bias/Adammain/q/dense/bias/Adam_1main/q/dense/kernelmain/q/dense/kernel/Adammain/q/dense/kernel/Adam_1main/q/dense_1/biasmain/q/dense_1/bias/Adammain/q/dense_1/bias/Adam_1main/q/dense_1/kernelmain/q/dense_1/kernel/Adammain/q/dense_1/kernel/Adam_1main/q/dense_2/biasmain/q/dense_2/bias/Adammain/q/dense_2/bias/Adam_1main/q/dense_2/kernelmain/q/dense_2/kernel/Adammain/q/dense_2/kernel/Adam_1target/pi/dense/biastarget/pi/dense/kerneltarget/pi/dense_1/biastarget/pi/dense_1/kerneltarget/pi/dense_2/biastarget/pi/dense_2/kerneltarget/q/dense/biastarget/q/dense/kerneltarget/q/dense_1/biastarget/q/dense_1/kerneltarget/q/dense_2/biastarget/q/dense_2/kernel*B
dtypes8
624

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 

+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
T0*

axis *
N*
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
T0*
_output_shapes
: 
Ż

save/RestoreV2/tensor_namesConst*

value
B
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel*
dtype0*
_output_shapes
:4
Ī
save/RestoreV2/shape_and_slicesConst*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:4

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*ę
_output_shapesÓ
Š::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624
£
save/AssignAssignbeta1_powersave/RestoreV2*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
: 
Ø
save/Assign_1Assignbeta1_power_1save/RestoreV2:1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*$
_class
loc:@main/q/dense/bias
§
save/Assign_2Assignbeta2_powersave/RestoreV2:2*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
: 
Ø
save/Assign_3Assignbeta2_power_1save/RestoreV2:3*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*$
_class
loc:@main/q/dense/bias
³
save/Assign_4Assignmain/pi/dense/biassave/RestoreV2:4*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ø
save/Assign_5Assignmain/pi/dense/bias/Adamsave/RestoreV2:5*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias
ŗ
save/Assign_6Assignmain/pi/dense/bias/Adam_1save/RestoreV2:6*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias
»
save/Assign_7Assignmain/pi/dense/kernelsave/RestoreV2:7*
T0*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
Ą
save/Assign_8Assignmain/pi/dense/kernel/Adamsave/RestoreV2:8*
T0*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
Ā
save/Assign_9Assignmain/pi/dense/kernel/Adam_1save/RestoreV2:9*
use_locking(*
T0*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
_output_shapes
:	
¹
save/Assign_10Assignmain/pi/dense_1/biassave/RestoreV2:10*
use_locking(*
T0*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
_output_shapes	
:¬
¾
save/Assign_11Assignmain/pi/dense_1/bias/Adamsave/RestoreV2:11*
validate_shape(*
_output_shapes	
:¬*
use_locking(*
T0*'
_class
loc:@main/pi/dense_1/bias
Ą
save/Assign_12Assignmain/pi/dense_1/bias/Adam_1save/RestoreV2:12*
use_locking(*
T0*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
_output_shapes	
:¬
Ā
save/Assign_13Assignmain/pi/dense_1/kernelsave/RestoreV2:13*
use_locking(*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
¬
Ē
save/Assign_14Assignmain/pi/dense_1/kernel/Adamsave/RestoreV2:14*
use_locking(*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
¬
É
save/Assign_15Assignmain/pi/dense_1/kernel/Adam_1save/RestoreV2:15*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
¬*
use_locking(
ø
save/Assign_16Assignmain/pi/dense_2/biassave/RestoreV2:16*
T0*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
½
save/Assign_17Assignmain/pi/dense_2/bias/Adamsave/RestoreV2:17*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
_output_shapes
:
æ
save/Assign_18Assignmain/pi/dense_2/bias/Adam_1save/RestoreV2:18*
T0*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Į
save/Assign_19Assignmain/pi/dense_2/kernelsave/RestoreV2:19*
use_locking(*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	¬
Ę
save/Assign_20Assignmain/pi/dense_2/kernel/Adamsave/RestoreV2:20*
use_locking(*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	¬
Č
save/Assign_21Assignmain/pi/dense_2/kernel/Adam_1save/RestoreV2:21*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	¬*
use_locking(
³
save/Assign_22Assignmain/q/dense/biassave/RestoreV2:22*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*$
_class
loc:@main/q/dense/bias
ø
save/Assign_23Assignmain/q/dense/bias/Adamsave/RestoreV2:23*
use_locking(*
T0*$
_class
loc:@main/q/dense/bias*
validate_shape(*
_output_shapes	
:
ŗ
save/Assign_24Assignmain/q/dense/bias/Adam_1save/RestoreV2:24*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*$
_class
loc:@main/q/dense/bias
»
save/Assign_25Assignmain/q/dense/kernelsave/RestoreV2:25*
use_locking(*
T0*&
_class
loc:@main/q/dense/kernel*
validate_shape(*
_output_shapes
:	
Ą
save/Assign_26Assignmain/q/dense/kernel/Adamsave/RestoreV2:26*
use_locking(*
T0*&
_class
loc:@main/q/dense/kernel*
validate_shape(*
_output_shapes
:	
Ā
save/Assign_27Assignmain/q/dense/kernel/Adam_1save/RestoreV2:27*
T0*&
_class
loc:@main/q/dense/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
·
save/Assign_28Assignmain/q/dense_1/biassave/RestoreV2:28*
use_locking(*
T0*&
_class
loc:@main/q/dense_1/bias*
validate_shape(*
_output_shapes	
:¬
¼
save/Assign_29Assignmain/q/dense_1/bias/Adamsave/RestoreV2:29*
use_locking(*
T0*&
_class
loc:@main/q/dense_1/bias*
validate_shape(*
_output_shapes	
:¬
¾
save/Assign_30Assignmain/q/dense_1/bias/Adam_1save/RestoreV2:30*
use_locking(*
T0*&
_class
loc:@main/q/dense_1/bias*
validate_shape(*
_output_shapes	
:¬
Ą
save/Assign_31Assignmain/q/dense_1/kernelsave/RestoreV2:31*
use_locking(*
T0*(
_class
loc:@main/q/dense_1/kernel*
validate_shape(* 
_output_shapes
:
¬
Å
save/Assign_32Assignmain/q/dense_1/kernel/Adamsave/RestoreV2:32*
use_locking(*
T0*(
_class
loc:@main/q/dense_1/kernel*
validate_shape(* 
_output_shapes
:
¬
Ē
save/Assign_33Assignmain/q/dense_1/kernel/Adam_1save/RestoreV2:33*
use_locking(*
T0*(
_class
loc:@main/q/dense_1/kernel*
validate_shape(* 
_output_shapes
:
¬
¶
save/Assign_34Assignmain/q/dense_2/biassave/RestoreV2:34*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*&
_class
loc:@main/q/dense_2/bias
»
save/Assign_35Assignmain/q/dense_2/bias/Adamsave/RestoreV2:35*
use_locking(*
T0*&
_class
loc:@main/q/dense_2/bias*
validate_shape(*
_output_shapes
:
½
save/Assign_36Assignmain/q/dense_2/bias/Adam_1save/RestoreV2:36*
use_locking(*
T0*&
_class
loc:@main/q/dense_2/bias*
validate_shape(*
_output_shapes
:
æ
save/Assign_37Assignmain/q/dense_2/kernelsave/RestoreV2:37*
use_locking(*
T0*(
_class
loc:@main/q/dense_2/kernel*
validate_shape(*
_output_shapes
:	¬
Ä
save/Assign_38Assignmain/q/dense_2/kernel/Adamsave/RestoreV2:38*
validate_shape(*
_output_shapes
:	¬*
use_locking(*
T0*(
_class
loc:@main/q/dense_2/kernel
Ę
save/Assign_39Assignmain/q/dense_2/kernel/Adam_1save/RestoreV2:39*
use_locking(*
T0*(
_class
loc:@main/q/dense_2/kernel*
validate_shape(*
_output_shapes
:	¬
¹
save/Assign_40Assigntarget/pi/dense/biassave/RestoreV2:40*
use_locking(*
T0*'
_class
loc:@target/pi/dense/bias*
validate_shape(*
_output_shapes	
:
Į
save/Assign_41Assigntarget/pi/dense/kernelsave/RestoreV2:41*
T0*)
_class
loc:@target/pi/dense/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
½
save/Assign_42Assigntarget/pi/dense_1/biassave/RestoreV2:42*
validate_shape(*
_output_shapes	
:¬*
use_locking(*
T0*)
_class
loc:@target/pi/dense_1/bias
Ę
save/Assign_43Assigntarget/pi/dense_1/kernelsave/RestoreV2:43*
validate_shape(* 
_output_shapes
:
¬*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_1/kernel
¼
save/Assign_44Assigntarget/pi/dense_2/biassave/RestoreV2:44*
use_locking(*
T0*)
_class
loc:@target/pi/dense_2/bias*
validate_shape(*
_output_shapes
:
Å
save/Assign_45Assigntarget/pi/dense_2/kernelsave/RestoreV2:45*
validate_shape(*
_output_shapes
:	¬*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_2/kernel
·
save/Assign_46Assigntarget/q/dense/biassave/RestoreV2:46*
use_locking(*
T0*&
_class
loc:@target/q/dense/bias*
validate_shape(*
_output_shapes	
:
æ
save/Assign_47Assigntarget/q/dense/kernelsave/RestoreV2:47*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*(
_class
loc:@target/q/dense/kernel
»
save/Assign_48Assigntarget/q/dense_1/biassave/RestoreV2:48*
use_locking(*
T0*(
_class
loc:@target/q/dense_1/bias*
validate_shape(*
_output_shapes	
:¬
Ä
save/Assign_49Assigntarget/q/dense_1/kernelsave/RestoreV2:49*
T0**
_class 
loc:@target/q/dense_1/kernel*
validate_shape(* 
_output_shapes
:
¬*
use_locking(
ŗ
save/Assign_50Assigntarget/q/dense_2/biassave/RestoreV2:50*
use_locking(*
T0*(
_class
loc:@target/q/dense_2/bias*
validate_shape(*
_output_shapes
:
Ć
save/Assign_51Assigntarget/q/dense_2/kernelsave/RestoreV2:51*
T0**
_class 
loc:@target/q/dense_2/kernel*
validate_shape(*
_output_shapes
:	¬*
use_locking(

save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard
[
save_1/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
dtype0*
_output_shapes
: *
shape: 

save_1/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_db8ea94323544206a603c21b4335aa15/part
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_1/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
^
save_1/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
Ü

save_1/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:4*

value
B
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel
Ķ
save_1/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:4*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
§
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1main/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1main/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1main/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1main/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1main/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1main/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1main/q/dense/biasmain/q/dense/bias/Adammain/q/dense/bias/Adam_1main/q/dense/kernelmain/q/dense/kernel/Adammain/q/dense/kernel/Adam_1main/q/dense_1/biasmain/q/dense_1/bias/Adammain/q/dense_1/bias/Adam_1main/q/dense_1/kernelmain/q/dense_1/kernel/Adammain/q/dense_1/kernel/Adam_1main/q/dense_2/biasmain/q/dense_2/bias/Adammain/q/dense_2/bias/Adam_1main/q/dense_2/kernelmain/q/dense_2/kernel/Adammain/q/dense_2/kernel/Adam_1target/pi/dense/biastarget/pi/dense/kerneltarget/pi/dense_1/biastarget/pi/dense_1/kerneltarget/pi/dense_2/biastarget/pi/dense_2/kerneltarget/q/dense/biastarget/q/dense/kerneltarget/q/dense_1/biastarget/q/dense_1/kerneltarget/q/dense_2/biastarget/q/dense_2/kernel*B
dtypes8
624

save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
£
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*
T0*

axis *
N*
_output_shapes
:

save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(

save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
T0*
_output_shapes
: 
ß

save_1/RestoreV2/tensor_namesConst*

value
B
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel*
dtype0*
_output_shapes
:4
Š
!save_1/RestoreV2/shape_and_slicesConst*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:4

save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*B
dtypes8
624*ę
_output_shapesÓ
Š::::::::::::::::::::::::::::::::::::::::::::::::::::
§
save_1/AssignAssignbeta1_powersave_1/RestoreV2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias
¬
save_1/Assign_1Assignbeta1_power_1save_1/RestoreV2:1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*$
_class
loc:@main/q/dense/bias
«
save_1/Assign_2Assignbeta2_powersave_1/RestoreV2:2*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
: 
¬
save_1/Assign_3Assignbeta2_power_1save_1/RestoreV2:3*
use_locking(*
T0*$
_class
loc:@main/q/dense/bias*
validate_shape(*
_output_shapes
: 
·
save_1/Assign_4Assignmain/pi/dense/biassave_1/RestoreV2:4*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes	
:
¼
save_1/Assign_5Assignmain/pi/dense/bias/Adamsave_1/RestoreV2:5*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes	
:
¾
save_1/Assign_6Assignmain/pi/dense/bias/Adam_1save_1/RestoreV2:6*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias
æ
save_1/Assign_7Assignmain/pi/dense/kernelsave_1/RestoreV2:7*
T0*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
Ä
save_1/Assign_8Assignmain/pi/dense/kernel/Adamsave_1/RestoreV2:8*
use_locking(*
T0*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
_output_shapes
:	
Ę
save_1/Assign_9Assignmain/pi/dense/kernel/Adam_1save_1/RestoreV2:9*
use_locking(*
T0*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
_output_shapes
:	
½
save_1/Assign_10Assignmain/pi/dense_1/biassave_1/RestoreV2:10*
use_locking(*
T0*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
_output_shapes	
:¬
Ā
save_1/Assign_11Assignmain/pi/dense_1/bias/Adamsave_1/RestoreV2:11*
validate_shape(*
_output_shapes	
:¬*
use_locking(*
T0*'
_class
loc:@main/pi/dense_1/bias
Ä
save_1/Assign_12Assignmain/pi/dense_1/bias/Adam_1save_1/RestoreV2:12*
use_locking(*
T0*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
_output_shapes	
:¬
Ę
save_1/Assign_13Assignmain/pi/dense_1/kernelsave_1/RestoreV2:13*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
¬*
use_locking(
Ė
save_1/Assign_14Assignmain/pi/dense_1/kernel/Adamsave_1/RestoreV2:14*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
¬*
use_locking(
Ķ
save_1/Assign_15Assignmain/pi/dense_1/kernel/Adam_1save_1/RestoreV2:15*
use_locking(*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
¬
¼
save_1/Assign_16Assignmain/pi/dense_2/biassave_1/RestoreV2:16*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias
Į
save_1/Assign_17Assignmain/pi/dense_2/bias/Adamsave_1/RestoreV2:17*
T0*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Ć
save_1/Assign_18Assignmain/pi/dense_2/bias/Adam_1save_1/RestoreV2:18*
T0*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Å
save_1/Assign_19Assignmain/pi/dense_2/kernelsave_1/RestoreV2:19*
use_locking(*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	¬
Ź
save_1/Assign_20Assignmain/pi/dense_2/kernel/Adamsave_1/RestoreV2:20*
use_locking(*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	¬
Ģ
save_1/Assign_21Assignmain/pi/dense_2/kernel/Adam_1save_1/RestoreV2:21*
use_locking(*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	¬
·
save_1/Assign_22Assignmain/q/dense/biassave_1/RestoreV2:22*
use_locking(*
T0*$
_class
loc:@main/q/dense/bias*
validate_shape(*
_output_shapes	
:
¼
save_1/Assign_23Assignmain/q/dense/bias/Adamsave_1/RestoreV2:23*
use_locking(*
T0*$
_class
loc:@main/q/dense/bias*
validate_shape(*
_output_shapes	
:
¾
save_1/Assign_24Assignmain/q/dense/bias/Adam_1save_1/RestoreV2:24*
use_locking(*
T0*$
_class
loc:@main/q/dense/bias*
validate_shape(*
_output_shapes	
:
æ
save_1/Assign_25Assignmain/q/dense/kernelsave_1/RestoreV2:25*
use_locking(*
T0*&
_class
loc:@main/q/dense/kernel*
validate_shape(*
_output_shapes
:	
Ä
save_1/Assign_26Assignmain/q/dense/kernel/Adamsave_1/RestoreV2:26*
use_locking(*
T0*&
_class
loc:@main/q/dense/kernel*
validate_shape(*
_output_shapes
:	
Ę
save_1/Assign_27Assignmain/q/dense/kernel/Adam_1save_1/RestoreV2:27*
T0*&
_class
loc:@main/q/dense/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
»
save_1/Assign_28Assignmain/q/dense_1/biassave_1/RestoreV2:28*
use_locking(*
T0*&
_class
loc:@main/q/dense_1/bias*
validate_shape(*
_output_shapes	
:¬
Ą
save_1/Assign_29Assignmain/q/dense_1/bias/Adamsave_1/RestoreV2:29*
T0*&
_class
loc:@main/q/dense_1/bias*
validate_shape(*
_output_shapes	
:¬*
use_locking(
Ā
save_1/Assign_30Assignmain/q/dense_1/bias/Adam_1save_1/RestoreV2:30*
use_locking(*
T0*&
_class
loc:@main/q/dense_1/bias*
validate_shape(*
_output_shapes	
:¬
Ä
save_1/Assign_31Assignmain/q/dense_1/kernelsave_1/RestoreV2:31*
use_locking(*
T0*(
_class
loc:@main/q/dense_1/kernel*
validate_shape(* 
_output_shapes
:
¬
É
save_1/Assign_32Assignmain/q/dense_1/kernel/Adamsave_1/RestoreV2:32*
use_locking(*
T0*(
_class
loc:@main/q/dense_1/kernel*
validate_shape(* 
_output_shapes
:
¬
Ė
save_1/Assign_33Assignmain/q/dense_1/kernel/Adam_1save_1/RestoreV2:33*
use_locking(*
T0*(
_class
loc:@main/q/dense_1/kernel*
validate_shape(* 
_output_shapes
:
¬
ŗ
save_1/Assign_34Assignmain/q/dense_2/biassave_1/RestoreV2:34*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*&
_class
loc:@main/q/dense_2/bias
æ
save_1/Assign_35Assignmain/q/dense_2/bias/Adamsave_1/RestoreV2:35*
use_locking(*
T0*&
_class
loc:@main/q/dense_2/bias*
validate_shape(*
_output_shapes
:
Į
save_1/Assign_36Assignmain/q/dense_2/bias/Adam_1save_1/RestoreV2:36*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*&
_class
loc:@main/q/dense_2/bias
Ć
save_1/Assign_37Assignmain/q/dense_2/kernelsave_1/RestoreV2:37*
use_locking(*
T0*(
_class
loc:@main/q/dense_2/kernel*
validate_shape(*
_output_shapes
:	¬
Č
save_1/Assign_38Assignmain/q/dense_2/kernel/Adamsave_1/RestoreV2:38*
use_locking(*
T0*(
_class
loc:@main/q/dense_2/kernel*
validate_shape(*
_output_shapes
:	¬
Ź
save_1/Assign_39Assignmain/q/dense_2/kernel/Adam_1save_1/RestoreV2:39*
use_locking(*
T0*(
_class
loc:@main/q/dense_2/kernel*
validate_shape(*
_output_shapes
:	¬
½
save_1/Assign_40Assigntarget/pi/dense/biassave_1/RestoreV2:40*
use_locking(*
T0*'
_class
loc:@target/pi/dense/bias*
validate_shape(*
_output_shapes	
:
Å
save_1/Assign_41Assigntarget/pi/dense/kernelsave_1/RestoreV2:41*
use_locking(*
T0*)
_class
loc:@target/pi/dense/kernel*
validate_shape(*
_output_shapes
:	
Į
save_1/Assign_42Assigntarget/pi/dense_1/biassave_1/RestoreV2:42*
T0*)
_class
loc:@target/pi/dense_1/bias*
validate_shape(*
_output_shapes	
:¬*
use_locking(
Ź
save_1/Assign_43Assigntarget/pi/dense_1/kernelsave_1/RestoreV2:43*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
¬
Ą
save_1/Assign_44Assigntarget/pi/dense_2/biassave_1/RestoreV2:44*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*)
_class
loc:@target/pi/dense_2/bias
É
save_1/Assign_45Assigntarget/pi/dense_2/kernelsave_1/RestoreV2:45*
validate_shape(*
_output_shapes
:	¬*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_2/kernel
»
save_1/Assign_46Assigntarget/q/dense/biassave_1/RestoreV2:46*
T0*&
_class
loc:@target/q/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Ć
save_1/Assign_47Assigntarget/q/dense/kernelsave_1/RestoreV2:47*
T0*(
_class
loc:@target/q/dense/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
æ
save_1/Assign_48Assigntarget/q/dense_1/biassave_1/RestoreV2:48*
T0*(
_class
loc:@target/q/dense_1/bias*
validate_shape(*
_output_shapes	
:¬*
use_locking(
Č
save_1/Assign_49Assigntarget/q/dense_1/kernelsave_1/RestoreV2:49*
validate_shape(* 
_output_shapes
:
¬*
use_locking(*
T0**
_class 
loc:@target/q/dense_1/kernel
¾
save_1/Assign_50Assigntarget/q/dense_2/biassave_1/RestoreV2:50*
T0*(
_class
loc:@target/q/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Ē
save_1/Assign_51Assigntarget/q/dense_2/kernelsave_1/RestoreV2:51*
use_locking(*
T0**
_class 
loc:@target/q/dense_2/kernel*
validate_shape(*
_output_shapes
:	¬
ģ
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_44^save_1/Assign_45^save_1/Assign_46^save_1/Assign_47^save_1/Assign_48^save_1/Assign_49^save_1/Assign_5^save_1/Assign_50^save_1/Assign_51^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard
[
save_2/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
r
save_2/filenamePlaceholderWithDefaultsave_2/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_2/ConstPlaceholderWithDefaultsave_2/filename*
shape: *
dtype0*
_output_shapes
: 

save_2/StringJoin/inputs_1Const*<
value3B1 B+_temp_7e9f940ed5374fe4b008d3dc51f5c0b6/part*
dtype0*
_output_shapes
: 
{
save_2/StringJoin
StringJoinsave_2/Constsave_2/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_2/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_2/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_2/ShardedFilenameShardedFilenamesave_2/StringJoinsave_2/ShardedFilename/shardsave_2/num_shards*
_output_shapes
: 
Ü

save_2/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:4*

value
B
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel
Ķ
save_2/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:4*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
§
save_2/SaveV2SaveV2save_2/ShardedFilenamesave_2/SaveV2/tensor_namessave_2/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1main/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1main/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1main/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1main/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1main/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1main/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1main/q/dense/biasmain/q/dense/bias/Adammain/q/dense/bias/Adam_1main/q/dense/kernelmain/q/dense/kernel/Adammain/q/dense/kernel/Adam_1main/q/dense_1/biasmain/q/dense_1/bias/Adammain/q/dense_1/bias/Adam_1main/q/dense_1/kernelmain/q/dense_1/kernel/Adammain/q/dense_1/kernel/Adam_1main/q/dense_2/biasmain/q/dense_2/bias/Adammain/q/dense_2/bias/Adam_1main/q/dense_2/kernelmain/q/dense_2/kernel/Adammain/q/dense_2/kernel/Adam_1target/pi/dense/biastarget/pi/dense/kerneltarget/pi/dense_1/biastarget/pi/dense_1/kerneltarget/pi/dense_2/biastarget/pi/dense_2/kerneltarget/q/dense/biastarget/q/dense/kerneltarget/q/dense_1/biastarget/q/dense_1/kerneltarget/q/dense_2/biastarget/q/dense_2/kernel*B
dtypes8
624

save_2/control_dependencyIdentitysave_2/ShardedFilename^save_2/SaveV2*
T0*)
_class
loc:@save_2/ShardedFilename*
_output_shapes
: 
£
-save_2/MergeV2Checkpoints/checkpoint_prefixesPacksave_2/ShardedFilename^save_2/control_dependency*
N*
_output_shapes
:*
T0*

axis 

save_2/MergeV2CheckpointsMergeV2Checkpoints-save_2/MergeV2Checkpoints/checkpoint_prefixessave_2/Const*
delete_old_dirs(

save_2/IdentityIdentitysave_2/Const^save_2/MergeV2Checkpoints^save_2/control_dependency*
T0*
_output_shapes
: 
ß

save_2/RestoreV2/tensor_namesConst*

value
B
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel*
dtype0*
_output_shapes
:4
Š
!save_2/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:4*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 

save_2/RestoreV2	RestoreV2save_2/Constsave_2/RestoreV2/tensor_names!save_2/RestoreV2/shape_and_slices*ę
_output_shapesÓ
Š::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624
§
save_2/AssignAssignbeta1_powersave_2/RestoreV2*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
: 
¬
save_2/Assign_1Assignbeta1_power_1save_2/RestoreV2:1*
T0*$
_class
loc:@main/q/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
«
save_2/Assign_2Assignbeta2_powersave_2/RestoreV2:2*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
: 
¬
save_2/Assign_3Assignbeta2_power_1save_2/RestoreV2:3*
T0*$
_class
loc:@main/q/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
·
save_2/Assign_4Assignmain/pi/dense/biassave_2/RestoreV2:4*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes	
:
¼
save_2/Assign_5Assignmain/pi/dense/bias/Adamsave_2/RestoreV2:5*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias
¾
save_2/Assign_6Assignmain/pi/dense/bias/Adam_1save_2/RestoreV2:6*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes	
:
æ
save_2/Assign_7Assignmain/pi/dense/kernelsave_2/RestoreV2:7*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*'
_class
loc:@main/pi/dense/kernel
Ä
save_2/Assign_8Assignmain/pi/dense/kernel/Adamsave_2/RestoreV2:8*
use_locking(*
T0*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
_output_shapes
:	
Ę
save_2/Assign_9Assignmain/pi/dense/kernel/Adam_1save_2/RestoreV2:9*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*'
_class
loc:@main/pi/dense/kernel
½
save_2/Assign_10Assignmain/pi/dense_1/biassave_2/RestoreV2:10*
use_locking(*
T0*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
_output_shapes	
:¬
Ā
save_2/Assign_11Assignmain/pi/dense_1/bias/Adamsave_2/RestoreV2:11*
T0*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
_output_shapes	
:¬*
use_locking(
Ä
save_2/Assign_12Assignmain/pi/dense_1/bias/Adam_1save_2/RestoreV2:12*
validate_shape(*
_output_shapes	
:¬*
use_locking(*
T0*'
_class
loc:@main/pi/dense_1/bias
Ę
save_2/Assign_13Assignmain/pi/dense_1/kernelsave_2/RestoreV2:13*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
¬*
use_locking(
Ė
save_2/Assign_14Assignmain/pi/dense_1/kernel/Adamsave_2/RestoreV2:14*
use_locking(*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
¬
Ķ
save_2/Assign_15Assignmain/pi/dense_1/kernel/Adam_1save_2/RestoreV2:15*
use_locking(*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
¬
¼
save_2/Assign_16Assignmain/pi/dense_2/biassave_2/RestoreV2:16*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
_output_shapes
:
Į
save_2/Assign_17Assignmain/pi/dense_2/bias/Adamsave_2/RestoreV2:17*
T0*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Ć
save_2/Assign_18Assignmain/pi/dense_2/bias/Adam_1save_2/RestoreV2:18*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
_output_shapes
:
Å
save_2/Assign_19Assignmain/pi/dense_2/kernelsave_2/RestoreV2:19*
use_locking(*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	¬
Ź
save_2/Assign_20Assignmain/pi/dense_2/kernel/Adamsave_2/RestoreV2:20*
validate_shape(*
_output_shapes
:	¬*
use_locking(*
T0*)
_class
loc:@main/pi/dense_2/kernel
Ģ
save_2/Assign_21Assignmain/pi/dense_2/kernel/Adam_1save_2/RestoreV2:21*
use_locking(*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	¬
·
save_2/Assign_22Assignmain/q/dense/biassave_2/RestoreV2:22*
T0*$
_class
loc:@main/q/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
¼
save_2/Assign_23Assignmain/q/dense/bias/Adamsave_2/RestoreV2:23*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*$
_class
loc:@main/q/dense/bias
¾
save_2/Assign_24Assignmain/q/dense/bias/Adam_1save_2/RestoreV2:24*
T0*$
_class
loc:@main/q/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
æ
save_2/Assign_25Assignmain/q/dense/kernelsave_2/RestoreV2:25*
T0*&
_class
loc:@main/q/dense/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
Ä
save_2/Assign_26Assignmain/q/dense/kernel/Adamsave_2/RestoreV2:26*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*&
_class
loc:@main/q/dense/kernel
Ę
save_2/Assign_27Assignmain/q/dense/kernel/Adam_1save_2/RestoreV2:27*
use_locking(*
T0*&
_class
loc:@main/q/dense/kernel*
validate_shape(*
_output_shapes
:	
»
save_2/Assign_28Assignmain/q/dense_1/biassave_2/RestoreV2:28*
T0*&
_class
loc:@main/q/dense_1/bias*
validate_shape(*
_output_shapes	
:¬*
use_locking(
Ą
save_2/Assign_29Assignmain/q/dense_1/bias/Adamsave_2/RestoreV2:29*
use_locking(*
T0*&
_class
loc:@main/q/dense_1/bias*
validate_shape(*
_output_shapes	
:¬
Ā
save_2/Assign_30Assignmain/q/dense_1/bias/Adam_1save_2/RestoreV2:30*
use_locking(*
T0*&
_class
loc:@main/q/dense_1/bias*
validate_shape(*
_output_shapes	
:¬
Ä
save_2/Assign_31Assignmain/q/dense_1/kernelsave_2/RestoreV2:31*
T0*(
_class
loc:@main/q/dense_1/kernel*
validate_shape(* 
_output_shapes
:
¬*
use_locking(
É
save_2/Assign_32Assignmain/q/dense_1/kernel/Adamsave_2/RestoreV2:32*
use_locking(*
T0*(
_class
loc:@main/q/dense_1/kernel*
validate_shape(* 
_output_shapes
:
¬
Ė
save_2/Assign_33Assignmain/q/dense_1/kernel/Adam_1save_2/RestoreV2:33*
validate_shape(* 
_output_shapes
:
¬*
use_locking(*
T0*(
_class
loc:@main/q/dense_1/kernel
ŗ
save_2/Assign_34Assignmain/q/dense_2/biassave_2/RestoreV2:34*
T0*&
_class
loc:@main/q/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
æ
save_2/Assign_35Assignmain/q/dense_2/bias/Adamsave_2/RestoreV2:35*
T0*&
_class
loc:@main/q/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Į
save_2/Assign_36Assignmain/q/dense_2/bias/Adam_1save_2/RestoreV2:36*
T0*&
_class
loc:@main/q/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Ć
save_2/Assign_37Assignmain/q/dense_2/kernelsave_2/RestoreV2:37*
validate_shape(*
_output_shapes
:	¬*
use_locking(*
T0*(
_class
loc:@main/q/dense_2/kernel
Č
save_2/Assign_38Assignmain/q/dense_2/kernel/Adamsave_2/RestoreV2:38*
use_locking(*
T0*(
_class
loc:@main/q/dense_2/kernel*
validate_shape(*
_output_shapes
:	¬
Ź
save_2/Assign_39Assignmain/q/dense_2/kernel/Adam_1save_2/RestoreV2:39*
use_locking(*
T0*(
_class
loc:@main/q/dense_2/kernel*
validate_shape(*
_output_shapes
:	¬
½
save_2/Assign_40Assigntarget/pi/dense/biassave_2/RestoreV2:40*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@target/pi/dense/bias
Å
save_2/Assign_41Assigntarget/pi/dense/kernelsave_2/RestoreV2:41*
use_locking(*
T0*)
_class
loc:@target/pi/dense/kernel*
validate_shape(*
_output_shapes
:	
Į
save_2/Assign_42Assigntarget/pi/dense_1/biassave_2/RestoreV2:42*
use_locking(*
T0*)
_class
loc:@target/pi/dense_1/bias*
validate_shape(*
_output_shapes	
:¬
Ź
save_2/Assign_43Assigntarget/pi/dense_1/kernelsave_2/RestoreV2:43*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
¬
Ą
save_2/Assign_44Assigntarget/pi/dense_2/biassave_2/RestoreV2:44*
T0*)
_class
loc:@target/pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
É
save_2/Assign_45Assigntarget/pi/dense_2/kernelsave_2/RestoreV2:45*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	¬
»
save_2/Assign_46Assigntarget/q/dense/biassave_2/RestoreV2:46*
use_locking(*
T0*&
_class
loc:@target/q/dense/bias*
validate_shape(*
_output_shapes	
:
Ć
save_2/Assign_47Assigntarget/q/dense/kernelsave_2/RestoreV2:47*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*(
_class
loc:@target/q/dense/kernel
æ
save_2/Assign_48Assigntarget/q/dense_1/biassave_2/RestoreV2:48*
use_locking(*
T0*(
_class
loc:@target/q/dense_1/bias*
validate_shape(*
_output_shapes	
:¬
Č
save_2/Assign_49Assigntarget/q/dense_1/kernelsave_2/RestoreV2:49*
validate_shape(* 
_output_shapes
:
¬*
use_locking(*
T0**
_class 
loc:@target/q/dense_1/kernel
¾
save_2/Assign_50Assigntarget/q/dense_2/biassave_2/RestoreV2:50*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*(
_class
loc:@target/q/dense_2/bias
Ē
save_2/Assign_51Assigntarget/q/dense_2/kernelsave_2/RestoreV2:51*
use_locking(*
T0**
_class 
loc:@target/q/dense_2/kernel*
validate_shape(*
_output_shapes
:	¬
ģ
save_2/restore_shardNoOp^save_2/Assign^save_2/Assign_1^save_2/Assign_10^save_2/Assign_11^save_2/Assign_12^save_2/Assign_13^save_2/Assign_14^save_2/Assign_15^save_2/Assign_16^save_2/Assign_17^save_2/Assign_18^save_2/Assign_19^save_2/Assign_2^save_2/Assign_20^save_2/Assign_21^save_2/Assign_22^save_2/Assign_23^save_2/Assign_24^save_2/Assign_25^save_2/Assign_26^save_2/Assign_27^save_2/Assign_28^save_2/Assign_29^save_2/Assign_3^save_2/Assign_30^save_2/Assign_31^save_2/Assign_32^save_2/Assign_33^save_2/Assign_34^save_2/Assign_35^save_2/Assign_36^save_2/Assign_37^save_2/Assign_38^save_2/Assign_39^save_2/Assign_4^save_2/Assign_40^save_2/Assign_41^save_2/Assign_42^save_2/Assign_43^save_2/Assign_44^save_2/Assign_45^save_2/Assign_46^save_2/Assign_47^save_2/Assign_48^save_2/Assign_49^save_2/Assign_5^save_2/Assign_50^save_2/Assign_51^save_2/Assign_6^save_2/Assign_7^save_2/Assign_8^save_2/Assign_9
1
save_2/restore_allNoOp^save_2/restore_shard "B
save_2/Const:0save_2/Identity:0save_2/restore_all (5 @F8"
train_op

Adam
Adam_1"Ż8
	variablesĻ8Ģ8

main/pi/dense/kernel:0main/pi/dense/kernel/Assignmain/pi/dense/kernel/read:021main/pi/dense/kernel/Initializer/random_uniform:08
v
main/pi/dense/bias:0main/pi/dense/bias/Assignmain/pi/dense/bias/read:02&main/pi/dense/bias/Initializer/zeros:08

main/pi/dense_1/kernel:0main/pi/dense_1/kernel/Assignmain/pi/dense_1/kernel/read:023main/pi/dense_1/kernel/Initializer/random_uniform:08
~
main/pi/dense_1/bias:0main/pi/dense_1/bias/Assignmain/pi/dense_1/bias/read:02(main/pi/dense_1/bias/Initializer/zeros:08

main/pi/dense_2/kernel:0main/pi/dense_2/kernel/Assignmain/pi/dense_2/kernel/read:023main/pi/dense_2/kernel/Initializer/random_uniform:08
~
main/pi/dense_2/bias:0main/pi/dense_2/bias/Assignmain/pi/dense_2/bias/read:02(main/pi/dense_2/bias/Initializer/zeros:08

main/q/dense/kernel:0main/q/dense/kernel/Assignmain/q/dense/kernel/read:020main/q/dense/kernel/Initializer/random_uniform:08
r
main/q/dense/bias:0main/q/dense/bias/Assignmain/q/dense/bias/read:02%main/q/dense/bias/Initializer/zeros:08

main/q/dense_1/kernel:0main/q/dense_1/kernel/Assignmain/q/dense_1/kernel/read:022main/q/dense_1/kernel/Initializer/random_uniform:08
z
main/q/dense_1/bias:0main/q/dense_1/bias/Assignmain/q/dense_1/bias/read:02'main/q/dense_1/bias/Initializer/zeros:08

main/q/dense_2/kernel:0main/q/dense_2/kernel/Assignmain/q/dense_2/kernel/read:022main/q/dense_2/kernel/Initializer/random_uniform:08
z
main/q/dense_2/bias:0main/q/dense_2/bias/Assignmain/q/dense_2/bias/read:02'main/q/dense_2/bias/Initializer/zeros:08

target/pi/dense/kernel:0target/pi/dense/kernel/Assigntarget/pi/dense/kernel/read:023target/pi/dense/kernel/Initializer/random_uniform:08
~
target/pi/dense/bias:0target/pi/dense/bias/Assigntarget/pi/dense/bias/read:02(target/pi/dense/bias/Initializer/zeros:08

target/pi/dense_1/kernel:0target/pi/dense_1/kernel/Assigntarget/pi/dense_1/kernel/read:025target/pi/dense_1/kernel/Initializer/random_uniform:08

target/pi/dense_1/bias:0target/pi/dense_1/bias/Assigntarget/pi/dense_1/bias/read:02*target/pi/dense_1/bias/Initializer/zeros:08

target/pi/dense_2/kernel:0target/pi/dense_2/kernel/Assigntarget/pi/dense_2/kernel/read:025target/pi/dense_2/kernel/Initializer/random_uniform:08

target/pi/dense_2/bias:0target/pi/dense_2/bias/Assigntarget/pi/dense_2/bias/read:02*target/pi/dense_2/bias/Initializer/zeros:08

target/q/dense/kernel:0target/q/dense/kernel/Assigntarget/q/dense/kernel/read:022target/q/dense/kernel/Initializer/random_uniform:08
z
target/q/dense/bias:0target/q/dense/bias/Assigntarget/q/dense/bias/read:02'target/q/dense/bias/Initializer/zeros:08

target/q/dense_1/kernel:0target/q/dense_1/kernel/Assigntarget/q/dense_1/kernel/read:024target/q/dense_1/kernel/Initializer/random_uniform:08

target/q/dense_1/bias:0target/q/dense_1/bias/Assigntarget/q/dense_1/bias/read:02)target/q/dense_1/bias/Initializer/zeros:08

target/q/dense_2/kernel:0target/q/dense_2/kernel/Assigntarget/q/dense_2/kernel/read:024target/q/dense_2/kernel/Initializer/random_uniform:08

target/q/dense_2/bias:0target/q/dense_2/bias/Assigntarget/q/dense_2/bias/read:02)target/q/dense_2/bias/Initializer/zeros:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0

main/pi/dense/kernel/Adam:0 main/pi/dense/kernel/Adam/Assign main/pi/dense/kernel/Adam/read:02-main/pi/dense/kernel/Adam/Initializer/zeros:0

main/pi/dense/kernel/Adam_1:0"main/pi/dense/kernel/Adam_1/Assign"main/pi/dense/kernel/Adam_1/read:02/main/pi/dense/kernel/Adam_1/Initializer/zeros:0

main/pi/dense/bias/Adam:0main/pi/dense/bias/Adam/Assignmain/pi/dense/bias/Adam/read:02+main/pi/dense/bias/Adam/Initializer/zeros:0

main/pi/dense/bias/Adam_1:0 main/pi/dense/bias/Adam_1/Assign main/pi/dense/bias/Adam_1/read:02-main/pi/dense/bias/Adam_1/Initializer/zeros:0

main/pi/dense_1/kernel/Adam:0"main/pi/dense_1/kernel/Adam/Assign"main/pi/dense_1/kernel/Adam/read:02/main/pi/dense_1/kernel/Adam/Initializer/zeros:0
 
main/pi/dense_1/kernel/Adam_1:0$main/pi/dense_1/kernel/Adam_1/Assign$main/pi/dense_1/kernel/Adam_1/read:021main/pi/dense_1/kernel/Adam_1/Initializer/zeros:0

main/pi/dense_1/bias/Adam:0 main/pi/dense_1/bias/Adam/Assign main/pi/dense_1/bias/Adam/read:02-main/pi/dense_1/bias/Adam/Initializer/zeros:0

main/pi/dense_1/bias/Adam_1:0"main/pi/dense_1/bias/Adam_1/Assign"main/pi/dense_1/bias/Adam_1/read:02/main/pi/dense_1/bias/Adam_1/Initializer/zeros:0

main/pi/dense_2/kernel/Adam:0"main/pi/dense_2/kernel/Adam/Assign"main/pi/dense_2/kernel/Adam/read:02/main/pi/dense_2/kernel/Adam/Initializer/zeros:0
 
main/pi/dense_2/kernel/Adam_1:0$main/pi/dense_2/kernel/Adam_1/Assign$main/pi/dense_2/kernel/Adam_1/read:021main/pi/dense_2/kernel/Adam_1/Initializer/zeros:0

main/pi/dense_2/bias/Adam:0 main/pi/dense_2/bias/Adam/Assign main/pi/dense_2/bias/Adam/read:02-main/pi/dense_2/bias/Adam/Initializer/zeros:0

main/pi/dense_2/bias/Adam_1:0"main/pi/dense_2/bias/Adam_1/Assign"main/pi/dense_2/bias/Adam_1/read:02/main/pi/dense_2/bias/Adam_1/Initializer/zeros:0
\
beta1_power_1:0beta1_power_1/Assignbeta1_power_1/read:02beta1_power_1/initial_value:0
\
beta2_power_1:0beta2_power_1/Assignbeta2_power_1/read:02beta2_power_1/initial_value:0

main/q/dense/kernel/Adam:0main/q/dense/kernel/Adam/Assignmain/q/dense/kernel/Adam/read:02,main/q/dense/kernel/Adam/Initializer/zeros:0

main/q/dense/kernel/Adam_1:0!main/q/dense/kernel/Adam_1/Assign!main/q/dense/kernel/Adam_1/read:02.main/q/dense/kernel/Adam_1/Initializer/zeros:0

main/q/dense/bias/Adam:0main/q/dense/bias/Adam/Assignmain/q/dense/bias/Adam/read:02*main/q/dense/bias/Adam/Initializer/zeros:0

main/q/dense/bias/Adam_1:0main/q/dense/bias/Adam_1/Assignmain/q/dense/bias/Adam_1/read:02,main/q/dense/bias/Adam_1/Initializer/zeros:0

main/q/dense_1/kernel/Adam:0!main/q/dense_1/kernel/Adam/Assign!main/q/dense_1/kernel/Adam/read:02.main/q/dense_1/kernel/Adam/Initializer/zeros:0

main/q/dense_1/kernel/Adam_1:0#main/q/dense_1/kernel/Adam_1/Assign#main/q/dense_1/kernel/Adam_1/read:020main/q/dense_1/kernel/Adam_1/Initializer/zeros:0

main/q/dense_1/bias/Adam:0main/q/dense_1/bias/Adam/Assignmain/q/dense_1/bias/Adam/read:02,main/q/dense_1/bias/Adam/Initializer/zeros:0

main/q/dense_1/bias/Adam_1:0!main/q/dense_1/bias/Adam_1/Assign!main/q/dense_1/bias/Adam_1/read:02.main/q/dense_1/bias/Adam_1/Initializer/zeros:0

main/q/dense_2/kernel/Adam:0!main/q/dense_2/kernel/Adam/Assign!main/q/dense_2/kernel/Adam/read:02.main/q/dense_2/kernel/Adam/Initializer/zeros:0

main/q/dense_2/kernel/Adam_1:0#main/q/dense_2/kernel/Adam_1/Assign#main/q/dense_2/kernel/Adam_1/read:020main/q/dense_2/kernel/Adam_1/Initializer/zeros:0

main/q/dense_2/bias/Adam:0main/q/dense_2/bias/Adam/Assignmain/q/dense_2/bias/Adam/read:02,main/q/dense_2/bias/Adam/Initializer/zeros:0

main/q/dense_2/bias/Adam_1:0!main/q/dense_2/bias/Adam_1/Assign!main/q/dense_2/bias/Adam_1/read:02.main/q/dense_2/bias/Adam_1/Initializer/zeros:0"ē
trainable_variablesĻĢ

main/pi/dense/kernel:0main/pi/dense/kernel/Assignmain/pi/dense/kernel/read:021main/pi/dense/kernel/Initializer/random_uniform:08
v
main/pi/dense/bias:0main/pi/dense/bias/Assignmain/pi/dense/bias/read:02&main/pi/dense/bias/Initializer/zeros:08

main/pi/dense_1/kernel:0main/pi/dense_1/kernel/Assignmain/pi/dense_1/kernel/read:023main/pi/dense_1/kernel/Initializer/random_uniform:08
~
main/pi/dense_1/bias:0main/pi/dense_1/bias/Assignmain/pi/dense_1/bias/read:02(main/pi/dense_1/bias/Initializer/zeros:08

main/pi/dense_2/kernel:0main/pi/dense_2/kernel/Assignmain/pi/dense_2/kernel/read:023main/pi/dense_2/kernel/Initializer/random_uniform:08
~
main/pi/dense_2/bias:0main/pi/dense_2/bias/Assignmain/pi/dense_2/bias/read:02(main/pi/dense_2/bias/Initializer/zeros:08

main/q/dense/kernel:0main/q/dense/kernel/Assignmain/q/dense/kernel/read:020main/q/dense/kernel/Initializer/random_uniform:08
r
main/q/dense/bias:0main/q/dense/bias/Assignmain/q/dense/bias/read:02%main/q/dense/bias/Initializer/zeros:08

main/q/dense_1/kernel:0main/q/dense_1/kernel/Assignmain/q/dense_1/kernel/read:022main/q/dense_1/kernel/Initializer/random_uniform:08
z
main/q/dense_1/bias:0main/q/dense_1/bias/Assignmain/q/dense_1/bias/read:02'main/q/dense_1/bias/Initializer/zeros:08

main/q/dense_2/kernel:0main/q/dense_2/kernel/Assignmain/q/dense_2/kernel/read:022main/q/dense_2/kernel/Initializer/random_uniform:08
z
main/q/dense_2/bias:0main/q/dense_2/bias/Assignmain/q/dense_2/bias/read:02'main/q/dense_2/bias/Initializer/zeros:08

target/pi/dense/kernel:0target/pi/dense/kernel/Assigntarget/pi/dense/kernel/read:023target/pi/dense/kernel/Initializer/random_uniform:08
~
target/pi/dense/bias:0target/pi/dense/bias/Assigntarget/pi/dense/bias/read:02(target/pi/dense/bias/Initializer/zeros:08

target/pi/dense_1/kernel:0target/pi/dense_1/kernel/Assigntarget/pi/dense_1/kernel/read:025target/pi/dense_1/kernel/Initializer/random_uniform:08

target/pi/dense_1/bias:0target/pi/dense_1/bias/Assigntarget/pi/dense_1/bias/read:02*target/pi/dense_1/bias/Initializer/zeros:08

target/pi/dense_2/kernel:0target/pi/dense_2/kernel/Assigntarget/pi/dense_2/kernel/read:025target/pi/dense_2/kernel/Initializer/random_uniform:08

target/pi/dense_2/bias:0target/pi/dense_2/bias/Assigntarget/pi/dense_2/bias/read:02*target/pi/dense_2/bias/Initializer/zeros:08

target/q/dense/kernel:0target/q/dense/kernel/Assigntarget/q/dense/kernel/read:022target/q/dense/kernel/Initializer/random_uniform:08
z
target/q/dense/bias:0target/q/dense/bias/Assigntarget/q/dense/bias/read:02'target/q/dense/bias/Initializer/zeros:08

target/q/dense_1/kernel:0target/q/dense_1/kernel/Assigntarget/q/dense_1/kernel/read:024target/q/dense_1/kernel/Initializer/random_uniform:08

target/q/dense_1/bias:0target/q/dense_1/bias/Assigntarget/q/dense_1/bias/read:02)target/q/dense_1/bias/Initializer/zeros:08

target/q/dense_2/kernel:0target/q/dense_2/kernel/Assigntarget/q/dense_2/kernel/read:024target/q/dense_2/kernel/Initializer/random_uniform:08

target/q/dense_2/bias:0target/q/dense_2/bias/Assigntarget/q/dense_2/bias/read:02)target/q/dense_2/bias/Initializer/zeros:08*Ž
serving_defaultŹ
)
x$
Placeholder:0’’’’’’’’’
+
a&
Placeholder_1:0’’’’’’’’’(
q#
main/q/Squeeze:0’’’’’’’’’*
pi$
main/pi/mul:0’’’’’’’’’tensorflow/serving/predict