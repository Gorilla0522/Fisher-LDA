import numpy as np
import matplotlib.pyplot as plt

#定数、変数
mean1 = [3,1]
mean2 = [1,3]
cov1 = [[1,2],[2,5]]
cov2 = [[1,2],[2,5]]

data1 = np.random.multivariate_normal(mean1,cov1,size=200)
data2 = np.random.multivariate_normal(mean2,cov2,size=200)
data = np.concatenate([data1,data2])

fig, ax = plt.subplots()
ax.plot(data1[:,0],data1[:,1],'o',color="blue",label="data1")
ax.plot(data2[:,0],data2[:,1],'o',color="orange",label="data2")
ax.legend(fontsize=7)
ax.set_xlim(-4,10)
ax.set_ylim(-4,10)
plt.rcParams["figure.figsize"] = (6,6)
plt.show()



def PCA(data1,data2):
    #1:データ共分散行列を求める
    #2:最大固有値に対応する固有ベクトルを求める
    #3:その方向が主成分空間である
    xs = np.concatenate([data1[:,0],data2[:,0]])
    ys = np.concatenate([data1[:,1],data2[:,1]])
    "標準化"
    x_mean = xs.mean()
    y_mean = ys.mean()
    xs = xs - x_mean 
    ys = ys - y_mean
    ""
    coor = np.stack([xs,ys])
    covs = np.cov(coor)
    print("covs = ",covs)
    eigen_values, eigen_vectors = np.linalg.eig(covs)
    print("eigen_values = ",eigen_values)
    print("eigen_vectors = ",eigen_vectors.T)
    max_index = np.argmax(eigen_values)
    print("max_eigen_vector = ",eigen_vectors.T[max_index])
    return eigen_vectors.T[max_index], x_mean, y_mean

def draw_line(vector):
    xs = np.array([])
    ys = np.array([])
    for i in range(-200,200,1):
        xs = np.append(xs,vector[0]*(0.1*i))
        ys = np.append(ys,vector[1]*(0.1*i))
    return xs,ys




xs1,ys1 = draw_line(eigen_vector)
#xs2,ys2 = draw_line(eigen_vector[1])

fig, ax = plt.subplots()
ax.plot(data1[:,0],data1[:,1],'o',color="blue",label="data1")
ax.plot(data2[:,0],data2[:,1],'o',color="orange",label="data2")
ax.plot(xs1 + x_mean,ys1 + y_mean,color="red")
#ax.plot(xs2 + x_mean,ys2 + y_mean,color="green")
ax.legend(fontsize=8)
ax.set_xlim(-4,10)
ax.set_ylim(-4,10)
plt.rcParams["figure.figsize"] = (6,6)
plt.show()
# print(eigen_vector[1]/eigen_vector[0])  #"draw_line関数が正常に機能しているかの確認"
# print((ys1[40]-ys1[30])/(xs1[40]-xs1[30]))




#自分で実装したPCAが正常に動作しているかライブラリを用いた結果と比較して確かめる.
import sklearn 
from sklearn.decomposition import PCA 

pca = PCA()
pca.fit(data)

testcov = pca.get_covariance()
evals,evecs  = np.linalg.eig(testcov)
print(evals)
print(evecs.T)

test_maxindex = np.argmax(evals)
test_pvec = evecs.T[test_maxindex]
testx, testy = draw_line(test_pvec)

fig, ax = plt.subplots()
ax.plot(data1[:,0],data1[:,1],'o',color="blue",label="data1")
ax.plot(data2[:,0],data2[:,1],'o',color="orange",label="data2")
ax.plot(testx + x_mean,testy + y_mean,color="red")
ax.legend(fontsize=8)
ax.set_xlim(-4,10)
ax.set_ylim(-4,10)
plt.rcParams["figure.figsize"] = (6,6)
plt.show()



#------------------------------
def fisher_LDA(data1,data2):
    #1:各クラスの平均を求める
    #2:クラス内共分散行列を求める
    #3:それらから適切な重み（ベクトル）wをreturnする
    xs_1 = data1[:,0]
    ys_1 = data1[:,1]
    xs_2 = data2[:,0]
    ys_2 = data2[:,1]

    mean1 = np.array([xs_1.mean(),ys_1.mean()])
    mean2 = np.array([xs_2.mean(),ys_2.mean()])
    mean_vector = mean2 - mean1
    
    var1 = np.cov(np.stack([xs_1 - xs_1.mean(),ys_1 - ys_1.mean()]))
    var2 = np.cov(np.stack([xs_2 - xs_2.mean(),ys_2 - ys_2.mean()]))
    sw_inverse = np.linalg.inv(var1 + var2)
    w = sw_inverse.dot(mean_vector)
    return w,mean1,mean2


w,m1,m2 = fisher_LDA(data1,data2)
xs_plot,ys_plot = draw_line(w)

fig, ax = plt.subplots()
ax.plot(data1[:,0],data1[:,1],'o',color="blue",label="data1")
ax.plot(data2[:,0],data2[:,1],'o',color="orange",label="data2")
ax.plot(xs_plot + x_mean,ys_plot + y_mean,color="red")
ax.legend(fontsize=8)
ax.plot(m1[0],m1[1],"o",color="green")
ax.plot(m2[0],m2[1],"o",color="green")
ax.set_xlim(-4,10)
ax.set_ylim(-4,10)
plt.rcParams["figure.figsize"] = (6,6)
plt.show()
# print(w[1]/w[0])  #"draw_line関数が正常に機能しているかの確認"
# print((ys_plot[40]-ys_plot[30])/(xs_plot[40]-xs_plot[30]))


hist_PCA = np.array([])
hist_LDA1 = np.array([])
hist_LDA2 = np.array([])
for i in range(len(data)):
    hist_PCA = np.append(hist_PCA,eigen_vector.dot(data[i]))
for i in range(len(data1)):
    hist_LDA1 = np.append(hist_LDA1,w.dot(data1[i]))
    hist_LDA2 = np.append(hist_LDA2,w.dot(data2[i]))


fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax1.hist(hist_PCA)
ax2.hist(hist_LDA1)
ax2.hist(hist_LDA2)