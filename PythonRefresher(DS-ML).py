# NUMPY (Revise)

import numpy as np
arr=np.array([(1,2,3),(4,5,6)])
arr2=np.array([(2,3),(4,5),(6,7)])

#prints the array
print(arr)

#prints the size of array
print(arr.size)

#prints the element size
print(arr.itemsize)

#prints the datatype of array
print(arr.dtype)

#prints the no. of dimensions of array
print(arr.ndim)

#prints the shape of array
print(arr.shape)

arr=arr.reshape(3,2)
#prints the reshaped array
print(arr)

#slicing of element 
print(arr[2][1])

#slicing with start element and boundary terminaation (:)
print(arr[0:,1])

#prints the max/min elements in array
print(arr.max())
print(arr.min())

#prints the sum of elements in array
print(arr.sum())

#Basics arithmetic operations of 2 arrays
print(arr+arr2)
print(arr-arr2)
print(arr*arr2)
print(arr/arr2)

#Prints the multidimensional inputs into the single arrays
print(arr.ravel())

#prints square root and standard deviation of elements of array
print(np.sqrt(arr))
print(np.std(arr))

#prints exponential and logarithmic values of elements of array
print(np.exp(arr))
print(np.log(arr))

var1=np.linspace(10,50,10)
#prints the generated 10 numbers between 10 and 50 equally spaced
print(var1)

#------------------------------------------------------------------------------------------------------

# PANDAS (Revise)

import pandas as pd
Employee1={'number':[1,2,3,4,5], 'name':["aditya","aditi","aditi","aditwa","aaditya"], 'hourly salary':[1,2,5,20,10]}
Employee2={'number':[1,2,3,4,5], 'name':["aditya","aaditya","aditi","aditwa","aaditya"], 'hourly salary':[15,26,75,10,40]}
PersonalInfo={'annual salary':[100,200,300,400,500], 'height':[170,165,180,180,190], 'weight':[100,70,79,80,85]}

#Converts the above to DataFrame format
table1=pd.DataFrame(Employee1)   
table2=pd.DataFrame(Employee2)   
table3=pd.DataFrame(PersonalInfo)

print(table1)

#Prints the starting 3 entries  
print(table1.head(3))

#Prints the last 2 entries  
print(table1.tail(2))

#Merging databases and printing them
fusion=pd.merge(table1,table2)
print(fusion)

#Merging databases, keeping only names which are common, doesn't depend on others
fusion=pd.merge(table1,table2,on='name')
print(fusion)


#Joining Table1 and table3
fuse=table1.join(table3)
print(fuse)

#Reading CSV file (converting it into HTML file)
colour=pd.read_csv('C:\\Users\\.......')  # --> Location of the root added
colour.to_html("NiceColour")  #--> Giving name to HTML file (text file)

#-----------------------------------------------------------------------------------------------------

# SCIPY (Revise)

import scipy as sp
from scipy import integrate 
from scipy import cluster
from scipy import fft
from scipy import special

#To get info about certain subpackage
sp.info(fft)

#To get source code about certain subpackage
sp.source(cluster)

#Work under special package : kelvin function
fun1=special.kelvin(15)
print(fun1)

#Work under special package : xlogy, exp(10 power), sin/cos
fun2=special.xlogy(2,8)
fun3=special.exp10(5)
fun4=special.sindg(90)
fun5=special.cosdg(0)

print(fun2)
print(fun3)
print(fun4)
print(fun5)

#Integrate quad function
var1= lambda x: x**3
function1=integrate.quad(var1,0,6)  #Limit from 0 to 6
print(function1)

#Double integration
var2= lambda y, x: x* y**4
function2=integrate.dblquad(var2,0,6,lambda x:0, lambda x:1) #Double quad integrate function
print(function2)

#Converting array into its Fourier Transform
arr=np.array([(2,4,6),(1,3,5)])
trans=sp.fft(arr)
print(trans)

#importing linear algebra from scipy
from scipy import linalg

#Solving Linear algebra between array 1 and array 2
array1=np.array([(1,2),(3,4)])
array2=np.array([(5,6),(7,8)])
finalfunction=sp.linalg.solve(array1,array2)
print(finalfunction)

#Reverse Function of array1
print(sp.linalg.inv(array1))

#-----------------------------------------------------------------------------------------------------

# MATPLOTLIB (Revise) 

# Simple plot (x-axis vs y-axis) with labels 
import matplotlib.pyplot as plt
from matplotlib import style

style.use('bmh')

plt.plot([0,5,10,15,20],[10,20,30,40,50])
plt.title('Test')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.show()

#Style subpackage to design the plot
style.use('dark_background')

#Bar Chart
x=[2,4,6,8]
y=[8,10,12,14]
plt.bar(x,y)
plt.show()

#Histogram
numbers=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
jumps=[0,5,10,15,20]
plt.hist(numbers,jumps,histtype='step')
plt.show()

#Pie Chart
foods=['pizza','ice cream','burgers']
sales[20,10,30]
color=['red','blue','green']
plt.pie(sales,labels=food,colors=color)
plt.show()

#Scatter plot
plt.title('ScatterPlot')
plt.scatter(x,y)
plt.show()

#-------------------------------------------------------------------------------

# SEABORN (Revise)

import seaborn as sns
#Loading dataset
database=sns.load_dataset('dataset_name')
print(database)

#DistPlot
sns.distplot(database['column_name'])
plt.show()

#JointPlot
sns.jointPlot(x='tips',y='total_bill',data=database)
plt.show()

#Catplot (with any kind we want)
database2=sns.load_dataset('flights')
sns.catplot(x='month',y='passengers',data=database2,kind='violin')  # By default it is kind='strip', another type: box (with interquartile ranges)
plt.show()

#FacetGrid
graph=sns.FacetGrid(database2, col='Gender',hue='smoker')
graph.map(plt.scatter,'total_expenses','tax')
plt.show()

