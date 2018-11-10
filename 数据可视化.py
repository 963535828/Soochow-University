#!/usr/bin/env python
# coding: utf-8

# In[49]:


get_ipython().run_line_magic('matplotlib', 'inline')
#%matplotlib
import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(0,2*np.pi,100)
y=np.sin(x)
plt.plot(x,y)


# In[5]:


fig=plt.figure()
ax=fig.add_axes([0.1,0.1,0.8,0.8])
ax.plot(x,y)

# 这是面向对象思想的绘图


# In[7]:


x=np.arange(0.0,5.0,0.02)
y=np.exp(-x)*np.cos(2*np.pi*x)
plt.plot(x,y)
plt.grid(color='grey')


# In[15]:


# 以面向对象方式来写
fig=plt.figure()
ax=fig.add_axes([0.9,0.4,0.8,0.8])#左，下，宽，高
ax.grid(color='gray')
ax.plot(x,y)


# In[17]:


fig=plt.figure()
ax=fig.add_axes([0.1,0.1,0.8,0.8])
ax.grid(color='blue')
ax.plot(x,y)
ax.set_xlabel('x axis')
ax.set_xlim((-2,10))


# In[19]:


fig=plt.figure()
ax=fig.add_axes([0.1,0.1,0.8,0.8])
ax.grid(color="grey")
ax.set_xlabel('x axis')  #设置x轴的标题
ax.set_xlim((0,5)) #设置x轴的刻度范围
ax.set_xticks(np.linspace(0,5,11)) #设置x轴的刻线值


# In[22]:


ax=plt.axes()
ax.plot(np.random.rand(50))
ax.yaxis.set_major_locator(plt.NullLocator())
ax.xaxis.set_major_formatter(plt.NullFormatter())


# In[29]:


#绘制脸谱，将横纵坐标的刻度全部去掉，不显示
# !pip3 install -U scikit-learn
from sklearn.datasets import fetch_olivetti_faces
faces=fetch_olivetti_faces().images
fig,ax=plt.subplots(5,5,figsize=(5,5))  # subplots()，创建子坐标，即分图
fig.subplots_adjust(hspace=0,wspace=0)  # 调整各分图之间的距离
for i in range(5):
    for j in range(5):
        ax[i,j].xaxis.set_major_locator(plt.NullLocator())
        ax[i,j].yaxis.set_major_locator(plt.NullLocator())
        ax[i,j].imshow(faces[10*i+j],cmap='bone')


# In[14]:


# 绘制自由落体的函数图像，s = gt^2 / 2
from matplotlib.ticker import MultipleLocator,FormatStrFormatter
# 引入两个操作刻线和标示的对象
t=np.linspace(0,100,100)
s=9.8*np.power(t,2)/2
fig,ax=plt.subplots(figsize=(8,4))
ax.plot(t,s)#至此，不对坐标轴进行设置，可以看到一种图示，默认的

# 设置x, y轴的说明
ax.set_ylabel('displacement')
ax.set_xlim(0,100)
ax.set_xlabel('time')

# 设置x轴刻线和标示，即刻度
xmajor_locator=MultipleLocator(20) #x轴上的主刻线是20的倍数
xmajor_formatter=FormatStrFormatter('%1.1f') #标示的显示格式
xminor_locator=MultipleLocator(5)

ax.xaxis.set_major_locator(xmajor_locator)
ax.xaxis.set_major_formatter(xmajor_formatter)
ax.xaxis.set_minor_locator(xminor_locator)

# 设置y轴
ymajor_locator=MultipleLocator(10000) #y轴上的主刻线是10000的倍数
ymajor_formatter=FormatStrFormatter('%1.1f') #标示的显示格式
yminor_locator=MultipleLocator(5000)

ax.yaxis.set_major_locator(ymajor_locator)
ax.yaxis.set_major_formatter(ymajor_formatter)
ax.yaxis.set_minor_locator(yminor_locator)

# 设置网格
ax.grid(True,which='major')
ax.grid(True,which='minor')

for tick in ax.xaxis.get_major_ticks():#???
    tick.label1.set_fontsize(16)


# In[15]:


fig,ax=plt.subplots(3,3,sharex='col',sharey='row') #3*3的坐标排列


# In[19]:


# 在Axes对象中增加文本。(0.5, 0.5)标示位置，x=0,y=0是左下角; x=1, y=1是右上角
fig,ax=plt.subplots(3,3,sharex='col',sharey='row')
for i in range(3):
    for j in range(3):
        ax[i,j].text(0.5,0.5,str((i,j)),fontsize=18,ha='center')


# In[21]:


for i in range(1,7):
    plt.subplot(2,3,i)##nrow=2, ncols=3, plot_number=i
    plt.text(0.5,0.5,str((2,3,i)),fontsize=16,ha='center')


# In[22]:


# 与上面的代码等效
fig=plt.figure()
fig.subplots_adjust(hspace=0.4,wspace=0.4)
for i in range(1,7):
    ax=fig.add_subplot(2,3,i)
    ax.text(0.5,0.5,str((2,3,i)),fontsize=16,ha='center')


# In[24]:


# 也可以有不规则的布局
fig=plt.figure()
ax1=fig.add_axes([0.1,0.1,0.8,0.8])##[left, bottom, width, height]
ax2=fig.add_axes([0.6,0.5,0.2,0.3])


# In[28]:


# 绘制正弦和余弦曲线，设置不同的线形状  [-.][-] [:] [--]
x=np.linspace(0,2*np.pi,100)
plt.plot(x,np.sin(x),linestyle='-.',color='blue')
plt.plot(x,np.cos(x),linestyle=':',color='red')


# In[32]:


# 对某些坐标点设置标记
plt.plot(range(10),linestyle='-',marker='o',markersize=16,
        markerfacecolor='b',color='r')
plt.grid(True)


# In[37]:


plt.plot(range(10),'-Dr',markersize=16,
        markerfacecolor='r',markevery=[2,4,6],linewidth=6)#指定点x=[2,4,6]


# In[42]:


a=np.arange(0,3,0.02)
b=np.arange(0,3,0.02)
c=np.exp(a)
d=c[::-1]

line1,=plt.plot(a,c,'k--',label='Model')#得到一个曲线对象
line2=plt.plot(a,d,'r:',label='Data')[0]#效果同上，换一种形式
line3=plt.plot(a,c+d,'b-',label='Total')
plt.legend(loc=0)#自动生成图例


# In[44]:


a = np.arange(0, 3, 0.02)
b = np.arange(0, 3, 0.02)
c = np.exp(a)
d = c[::-1]

line1, = plt.plot(a, c, 'k--', label="Model")    #得到一个曲线对象
line2 = plt.plot(a, d, "r:", label='Data')[0]    #效果同上，换一种形式
line3 = plt.plot(a, c+d, 'b-', label="Total")

plt.legend((line1,line2),loc=0)

#生成图例，虽然有三条线，但是这里只生成两条新的图例。
#loc=0，意味着系统自动选择最佳位置。变换窗口大小，图例位置变化


# In[46]:


# 从数据集中读入数据
import pandas as pd
cities=pd.read_csv(r'E:\Python-github\DataSet\jiangsu\city_population.csv')
cities


# In[61]:


lat=cities['latd']
lon=cities['longd']
population=cities['population']
area=cities['area']

plt.scatter(lon,lat,label=None,
           c=np.log10(population),
           cmap='viridis',s=area,
           linewidth=0,alpha=0.5)

plt.axis(aspect='equal')
plt.xlabel('longitude')
plt.ylabel('latitude')

plt.colorbar(label='log$_{10}$(population)')
#绘制数据光谱，有的资料翻译为“彩色条”

for area in [100,300,500]:
    plt.scatter([],[],c='k',alpha=0.3,s=area,label=str(area)+'km$^2$')

plt.legend(scatterpoints=1,frameon=False,labelspacing=1,title='City Area')
plt.title('Jiangsu Cities:Area and Population')


# In[62]:


# 另外一种绘制散点图的方法，使用plt.plot()
x=np.linspace(-np.pi,np.pi,9)
plt.plot(x,np.cos(x),'Dr',markerfacecolor='b',markersize=12)
plt.grid(True)


# In[63]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=[2,10,4,8,6]
position=[1,2,3,4,5]
plt.bar(x=position,height=data)


# In[68]:


position=[1,2.5,3,4.5,5]
plt.bar(x=position,height=data,width=0.4
        ,bottom=[3,0,5,0,1])
plt.grid(1)


# In[74]:


# 装扮后的柱子
data=[2,10,4,8,6]
position=[1,2,3,4,5]
labels=['Beijing','Soochow','Shanghai','Hangzhou','Hongkang']
plt.bar(x=position,height=data,
       width=0.4,color='b',
       edgecolor='r',linestyle='--',
       linewidth=3,hatch='x',tick_label=labels)


# In[78]:


# 堆积柱形图
position=np.arange(1,6)
a=np.random.rand(5)
b=np.random.rand(5)
plt.bar(position,a,label='a',color='b')
plt.bar(position,b,bottom=a,label='b',color='r')
plt.legend(loc=0)


# In[81]:


# 簇状柱形图
position=np.arange(1,6)
a=np.random.rand(5)
b=np.random.rand(5)

total_width=0.8
n=2
width=total_width/n
position=position-(total_width-width)/n

plt.bar(position,a,width=width,label='a',color='b')
plt.bar(position+width,b,width=width,label='b',color='r')
plt.legend(loc=0)


# In[82]:


#都是正数的条形图
position = np.arange(1, 6)
a = np.random.random(5)
plt.barh(position,a)


# In[85]:


#正负条形图，即有正数和负数
position = np.arange(1, 6)
a = np.random.random(5)
b = np.random.random(5)
plt.barh(position,a,color='g',label='a')
plt.barh(position,-b,color='r',label='b')
plt.legend(loc=0)


# In[90]:


fig,ax=plt.subplots(1,2)
data=[1,5,9,2]
ax[0].boxplot([data])
ax[0].grid(1)

ax[1].boxplot([data],showmeans=1)#显示平均值
ax[1].grid(1)


# In[96]:


np.random.seed(12345)#随机数的种子，每次执行，随机数都一样
data=pd.DataFrame(np.random.rand(5,4),columns=['A',"B",'C','D'])
data.boxplot(sym='r*',vert=0,meanline=0,showmeans=1)


# In[104]:


x=[2,4,6,8]
fig,ax=plt.subplots()
labels=['A','B','C','D']
colors=['red','yellow','blue','green']
explode=(0,0.1,0,0)

ax.pie(x,explode=explode,labels=labels,colors=colors,
      autopct='%1.1f%%',shadow=1,startangle=90,radius=1.2)
ax.set(aspect='equal',title='Pie')


# In[109]:


#正态分布直方图
fig=plt.figure()
mu=100#平均值
sigma=15#标准差
x=mu+sigma*np.random.randn(10000)
num_bins=50
n,bins,patches=plt.hist(x,num_bins,
                       density=True,facecolor="blue",
                       alpha=0.5,color='r')


# In[113]:


#在上述直方图外围绘制正态分布曲线
import matplotlib.mlab as mlab
fig=plt.figure()
mu=100
sigma=15
x=mu+sigma*np.random.randn(10000)

mun_bins=50
n,bins,patches=plt.hist(x,num_bins,density=True,facecolor='blue',
                       alpha=0.5,color='r')
y=mlab.normpdf(bins,mu,sigma)
plt.plot(bins,y,'r--')


# In[116]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

fig=plt.figure()
ax=plt.axes(projection='3d')

x_line=np.linspace(0,15,1000)
y_line=np.sin(x_line)
z_line=np.cos(x_line)
ax.plot3D(x_line,y_line,z_line,'b')

x_point=15*np.random.random(100)
y_point=np.sin(x_point)+0.1*np.random.random(100)
z_point=np.cos(x_point)+0.1*np.random.random(100)
ax.scatter3D(x_point,y_point,z_point,c=x_point,cmap='Reds')


# In[129]:


#莫比乌斯带
u=np.linspace(0,2*np.pi,30)
v=np.linspace(-0.5,0.5,8)/2.0
v,u=np.meshgrid(v,u)

phi=0.5*u
phi
r=1+v*np.cos(phi)
x=np.ravel(r*np.cos(u)) #范围一维数组
y=np.ravel(r*np.sin(u))
z=np.ravel(v*np.sin(phi))

from matplotlib.tri import Triangulation
#matplotlib.tri专门针对非结构化网络作图的模块，Triangulation实现元素为三角形的非机构化网络
tri=Triangulation(np.ravel(v),np.ravel(u))

ax=plt.axes(projection='3d')
ax.plot_trisurf(x,y,z,triangles=tri.triangles,cmap='viridis',linewidths=0.2)

ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)


# In[42]:


get_ipython().run_line_magic('pinfo2', 'sns.load_dataset')


# In[2]:


#coding:utf-8
import seaborn as sns
import pandas as pd
#鸢尾花数据集
iris=sns.load_dataset("iris",engine='python')###
iris.head()
#iris.head()
#species:种类
#sepal_length:花萼长度
#sepal_width:花萼宽度
#petal_length:花瓣长度
#petal_width:花瓣宽度


# In[3]:


#利用sns中的swarmplot函数绘制图像
sns.swarmplot(x='species',y='petal_length',data=iris)


# In[4]:


#泰坦尼克的数据
titanic=sns.load_dataset('titanic',engine='python')
g=sns.barplot(x='class',y='survived',
             hue='sex',data=titanic)


# In[13]:


# 对比matplotlib和seaborn的绘图
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data=np.random.multivariate_normal([0,0],[[5,2],[2,2]],size=2000)
data=pd.DataFrame(data,columns=['x','y'])
#绘制通常的直方图
for col in 'xy':
    plt.hist(data[col],density=1,alpha=0.5)


# In[14]:


#利用seaborn绘制正太分布图
data=np.random.multivariate_normal([0,0],[[5,2],[2,2]],size=2000)
data=pd.DataFrame(data,columns=['x','y'])
for col in 'xy':
    sns.kdeplot(data[col],shade=1)


# In[16]:


#将柱状图和曲线结合的KDE（核密度）图像
sns.distplot(data['x'])
sns.distplot(data['y'])


# In[17]:


#单独绘制KDE的方法
sns.kdeplot(data)


# In[19]:


#设置背景为白色
with sns.axes_style('white'):
    sns.jointplot('x','y',data,kind='kde')#2维的核密度估计


# In[20]:


#kind的另外一个设置效果
with sns.axes_style('white'):
    sns.jointplot('x','y',data,kind='hex')


# In[5]:


# 了解鸢尾花中数据关系
iris=sns.load_dataset('iris',engine='python')
sns.pairplot(iris,hue='species',height=2.5)


# In[6]:


# 了解小费数据中各个特征之间的关系
tips=sns.load_dataset('tips',engine='python')
tips['tip_pct']=100*tips['tip']/tips['total_bill']
#小费占总价的百分比
tips.head()


# In[11]:


#FacetGrid类，设置多个子图，用于显示指定条件下的变量关系
#创建子图对象，但是里面还没有具体的关系图像
import numpy as np
import matplotlib.pyplot as plt
grid=sns.FacetGrid(tips,row='sex',col='time',
                  margin_titles=True)

#每个子图中的直方图
grid.map(plt.hist,'tip_pct',bins=np.linspace(0,40,15))


# In[12]:


#在子图中，也可以绘制其它的图像，比如散点图
g=sns.FacetGrid(tips,col='time',row='smoker')#注意这里的行列都换了
g.map(plt.scatter,'total_bill','tip',edgecolor='w')


# In[14]:


g=sns.FacetGrid(tips,col='time',hue='smoker')
g.map(plt.scatter,'total_bill','tip',
     edgecolor='w').add_legend()#增加图例，这样可以减少子图数量（放大原图看）


# In[18]:


#改变子图的高低和比例aspect，表示高宽比例
bins=np.arange(0,65,5)
g=sns.FacetGrid(tips,col='day',height=4,aspect=.5)
g.map(plt.hist,'total_bill',bins=bins)


# In[19]:


#特别指定图的顺序
g=sns.FacetGrid(tips,col='smoker',col_order=['Yes',"No"])
g.map(plt.hist,'total_bill',bins=bins,color='m')


# In[20]:


#指定图例中色彩顺序
kws=dict(s=50,linewidth=.5,edgecolor='w')
g=sns.FacetGrid(tips,col='sex',hue='time',palette='Set1',
                hue_order=['Dinner','Lunch'])
g.map(plt.scatter,'total_bill','tip',**kws).add_legend()
#注意观察图例中颜色顺序和上面规定的hue_order


# In[22]:


#或者在字典中指定某个特征的颜色
pal=dict(Lunch='seagreen',Dinner='gray')
g=sns.FacetGrid(tips,col='sex',hue='time',palette=pal,
               hue_order=['Dinner','Lunch'])
g.map(plt.scatter,'total_bill','tip',**kws).add_legend()


# In[23]:


#用不同的形状表示不同特征
g=sns.FacetGrid(tips,col='sex',hue='time',palette=pal,
               hue_order=['Dinner','Lunch'])
g.map(plt.scatter,'total_bill','tip',**kws).add_legend()


# In[27]:


#设置列的子图个数，然后自动进行排布

att = sns.load_dataset("attention",engine='python')    #一个新的数据集
g=sns.FacetGrid(att,col='subject',col_wrap=5,height=1.5) #子图为5列
g.map(plt.plot,'solutions','score',marker='.')


# In[29]:


from scipy import stats
# stats 包含了大量的概率分布函数，详细阅读：https://docs.scipy.org/doc/scipy/reference/stats.html
def qplot(x,y,**kwargs):
    _,xr=stats.probplot(x,fit=False)#xr，返回一个排序结果
    _,yr=stats.probplot(y,fit=False)
    plt.scatter(xr,yr,**kwargs)
g=sns.FacetGrid(tips,col='smoker',hue='sex')
g.map(qplot,'total_bill','tip',**kws).add_legend()


# In[33]:


with sns.axes_style(style='ticks'):#with，上下文管理器，以确定坐标样式
    g=sns.catplot('day','total_bill','sex',data=tips,kind='box')  #tips数据集
    g.set_axis_labels('Day','Total Bill')              
    


# In[31]:


with sns.axes_style('white'):
    sns.jointplot("total_bill", "tip", data=tips, kind='hex')


# In[32]:


sns.jointplot("total_bill", "tip", data=tips, kind="reg")


# In[ ]:




