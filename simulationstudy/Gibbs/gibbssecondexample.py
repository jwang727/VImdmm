import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from collections import Counter

import math
import csv

mcmcseed=1
rng = np.random.default_rng(mcmcseed)

from scipy.stats import multivariate_normal, invwishart, wishart, dirichlet, rv_discrete, beta, gamma

import sys
print(sys.version)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataseedtrain", type=int, required=True)
parser.add_argument("--N", type=int, required=True)
args = parser.parse_args()

dataseedtrain = args.dataseedtrain
N = args.N
print(f"Running with dataseedtrain = {dataseedtrain} and N = {N}")

param_rng = np.random.default_rng(seed=123456) 

def generate_data(N, K, high_prob, param_rng, data_seed):

    data_rng = np.random.default_rng(seed=data_seed) 

    cluster_probs = param_rng.uniform(1 / (2 * K), 2 / K, K)
    cluster_probs /= cluster_probs.sum()

    z = data_rng.choice(K, size=N, p=cluster_probs)

    columns = [f'catcol_{i+1}' for i in range(2*K)] 
    categoricaldata = pd.DataFrame(columns=columns)

    for i in range(N):
        row = [] 
        for j in range(K):
            if z[i] == j:

                row.extend(data_rng.choice([0, 1], 2, p=[1-high_prob, high_prob]))
            else:

                row.extend(data_rng.choice([0, 1], 2, p=[high_prob, 1-high_prob]))


        categoricaldata.loc[i] = row


    continuous_dim = K  
    continuousdata = np.zeros((N, continuous_dim))
    meancol=np.zeros((N, continuous_dim))
    covdiagcol=np.zeros((N, continuous_dim))

    for i in range(N):
        cluster = z[i]
        mean = np.zeros(continuous_dim)
        mean[cluster] = 1  

        cov_matrix = 4 * np.eye(continuous_dim)
        cov_matrix[cluster, cluster] = 9  

        continuousdata[i] = data_rng.multivariate_normal(mean, cov_matrix)

        meancol[i]=mean
        covdiagcol[i]=np.diag(cov_matrix)


    data = pd.DataFrame(categoricaldata) 
    for j in range(0,continuous_dim):
        data[f"contcol_{j+1}"] = continuousdata[:, j]


    data['clusterlabel'] = z  

    return data, continuousdata, categoricaldata, z, cluster_probs, meancol, covdiagcol 


Ktrue=5
q=Ktrue
p=2*Ktrue
categorysizes=[1] 
highprob=0.9
d_1=2

truemu_kg=np.zeros([Ktrue,1,q])

for k in range(0,Ktrue):
    print('k is '+str(k))
    truemu_kg[k,0,k]=1


trueSigma_k=np.zeros([Ktrue,1,q,q])

for k in range(0,Ktrue):
    trueSigma_k[k,0,:]=4 * np.eye(q)
    trueSigma_k[k,0,k,k]=9


truepsi=np.zeros([Ktrue,p,2]) 

for k in range(Ktrue):
        for j in range(p):
            if j in range(2 * k, min(2 * k + 2, p)):

                truepsi[k, j, 1] = highprob
                truepsi[k, j, 0] = 1-highprob
            else:

                truepsi[k, j, 1] = 1-highprob
                truepsi[k, j, 0] = highprob


fulldata, continuousdata, categoricaldata, component, cluster_probs, meancol, covdiagcol=generate_data(N,Ktrue,0.9, param_rng, dataseedtrain)

trueclusterprob=cluster_probs

contcol_list = [f'contcol_{i}' for i in range(1, q+1)]
catcol_list=[f'catcol_{i}' for i in range(1,p+1)]

discrete_cols=catcol_list+['clusterlabel']

for col in discrete_cols:
    fulldata[col] = fulldata[col].astype(int)


allcategoricalvalues=[0,1]

modelled_continuous_columns=contcol_list 
modelled_categorical_columns=catcol_list

def merge_columns(row):
    return '_'.join(row.astype(str))


fulldata['combinedcategories'] = fulldata[modelled_categorical_columns].apply(merge_columns, axis=1)

modelled_categorical_columns_combined=['combinedcategories']

fulldataframe=pd.DataFrame(fulldata, columns = modelled_continuous_columns+modelled_categorical_columns+modelled_categorical_columns_combined+['clusterlabel'])

scaler = StandardScaler()

fulldataframe[modelled_continuous_columns] = scaler.fit_transform(fulldataframe[modelled_continuous_columns])

scaler.fit_transform(np.array(fulldata[modelled_continuous_columns])) 

combinedcategories=np.array(fulldataframe['combinedcategories'])

continuousdata=np.array(fulldataframe[modelled_continuous_columns])

from itertools import product

def generate_tuple(p):

    binary_combinations = ['_'.join(map(str, comb)) for comb in product([0, 1], repeat=p)]

    binary_array = np.array(binary_combinations)

    zeros_array = np.zeros(len(binary_combinations), dtype=int)

    return (binary_array, zeros_array)


allcatvals=generate_tuple(p)

fulldataframegroupbymeans = fulldataframe.groupby(['combinedcategories'], as_index=False)[modelled_continuous_columns].mean()

fulldataframegroupbystd = fulldataframe.groupby(['combinedcategories'], as_index=False)[modelled_continuous_columns].std()

categorical_columns_dims=[]

for catcols in modelled_categorical_columns:
    categorical_columns_dims.append(len(fulldataframe[catcols].unique()))


print('done simulating data')

print('check true parameter values')

print('truemu_kg')
print(truemu_kg)

print('trueSigma_kg')
print(trueSigma_k)

print('truepsi')
print(truepsi)

print('trueclusterprob')
print(trueclusterprob)

Ntest=min(int(N*0.4),2000)

testdata, testcontinuousdata, testcategoricaldata, testz, testcluster_probs, testmeancol, testcovdiagcol=generate_data(Ntest,Ktrue,0.9,param_rng,dataseedtrain+1000)

testdata['combinedcategories'] = testdata[modelled_categorical_columns].apply(merge_columns, axis=1)

testdataframe=pd.DataFrame(testdata, columns = modelled_continuous_columns+modelled_categorical_columns+modelled_categorical_columns_combined+['clusterlabel'])

scalertest = StandardScaler()

testdataframe[modelled_continuous_columns] = scalertest.fit_transform(testdataframe[modelled_continuous_columns])

testcontinuousdata=np.array(testdataframe[modelled_continuous_columns])

import pickle

outputtruemu_kg = open('truemu_kg'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(truemu_kg, outputtruemu_kg)
outputtruemu_kg.close()

outputtrueSigma_k = open('trueSigma_k'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(trueSigma_k, outputtrueSigma_k)
outputtrueSigma_k.close()

outputtruepsi = open('truepsi'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(truepsi, outputtruepsi)
outputtruepsi.close()

outputtrueclusterprob = open('trueclusterprob'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(trueclusterprob, outputtrueclusterprob)
outputtrueclusterprob.close()

outputfulldata=open('trainingdata'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl','wb')
pickle.dump(fulldata,outputfulldata)
outputfulldata.close()

outputtestdata=open('testdata'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl','wb')
pickle.dump(testdata,outputtestdata)
outputtestdata.close()

print('done saving true params')

Psiindependentofk=0

catvaluetoindex = dict(zip(allcatvals[0], list(range(0, len(allcatvals[0])))))

mudependentong=0
Sigmadependentong=0
pidependentong=0

if mudependentong==0:
    catvaluetoindexmu = dict(zip(allcatvals[0], [0] * len(allcatvals[0]) )) 

    allcatvalsmu=np.array(['0'])

    catvaluetoindexmu['0']=0
    combinedcategoriesmu=np.array(['0']*N)
else:
    allcatvalsmu=allcatvals
    catvaluetoindexmu = dict(zip(allcatvals[0], list(range(0, len(allcatvals[0])))))
    combinedcategoriesmu=combinedcategories

if Sigmadependentong==0:
    catvaluetoindexSigma = dict(zip(allcatvals[0], [0] * len(allcatvals[0]) ))
    allcatvalsSigma=np.array(['0'])

    catvaluetoindexSigma['0']=0
    combinedcategoriesSigma=np.array(['0']*N)
else:
    allcatvalsSigma=allcatvals
    catvaluetoindexSigma = dict(zip(allcatvals[0], list(range(0, len(allcatvals[0])))))
    combinedcategoriesSigma=combinedcategories

if pidependentong==0:
    catvaluetoindexpi = dict(zip(allcatvals[0], [0] * len(allcatvals[0]) ))
    allcatvalspi=np.array(['0'])

    catvaluetoindexpi['0']=0
    combinedcategoriespi=np.array(['0']*N)
else:
    allcatvalspi=allcatvals
    catvaluetoindexpi = dict(zip(allcatvals[0], list(range(0, len(allcatvals[0])))))
    combinedcategoriespi=combinedcategories

print('check mu catval variables')
print(allcatvalsmu)
print(catvaluetoindexmu)
print(Counter(combinedcategoriesmu))

print('check Sigma catval variables')
print(allcatvalsSigma)
print(catvaluetoindexSigma)
print(Counter(combinedcategoriesSigma))

print('check pi catval variables')
print(allcatvalspi)
print(catvaluetoindexpi)
print(Counter(combinedcategoriespi))

from scipy.stats import multivariate_normal, invwishart, wishart, dirichlet, rv_discrete, beta, gamma

K = Ktrue
p = len(modelled_categorical_columns)
q = len(modelled_continuous_columns) 

checkrow = 3
print(combinedcategories[checkrow])
print(catvaluetoindex[combinedcategories[checkrow]])

iterations = 5001 
burn_in = int(iterations * 0.2)

mdims = tuple([iterations,K] + [len(allcatvalsmu[0])] + [q])
betadims=tuple([iterations,K] + [len(allcatvalsmu[0])])

beta_kg_prior=1.0

z = np.zeros((N, 1))

zprobnormalised = np.zeros((N, K))+1/K

from kmodes.kprototypes import KPrototypes

cat_indices = [fulldataframe[modelled_continuous_columns+modelled_categorical_columns].columns.get_loc(col) for col in modelled_categorical_columns] 

kproto = KPrototypes(n_clusters=K, init='Cao', n_jobs=-1, random_state=42)
kprotoclusters = kproto.fit_predict(fulldataframe[modelled_continuous_columns+modelled_categorical_columns].to_numpy(), categorical=cat_indices)

z[:,0]=kprotoclusters

z_history = np.zeros((iterations, N), dtype=np.int8 if K < 128 else np.int16)
z_history[0] = z[:, 0]        

mu = np.zeros(mdims)  

print('check mu syntax')

Sigma = np.zeros((iterations, K, len(allcatvalsSigma[0]), q, q))

piprobdim = tuple([iterations]+ [K])

piprob = np.zeros(piprobdim)  

Psidims = tuple([iterations, K, p] + [np.max(categorical_columns_dims)])

print('check Psidims')
print(Psidims)

Psi = np.zeros(Psidims)  

Phi = 0.25*np.eye(q) 
nu = q + K + 1

mu_cs0_all = np.zeros((tuple([len(allcatvalsmu[0])] + [continuousdata.shape[1]])))

print(np.shape(mu_cs0_all))

if mudependentong==0:
    for j in allcatvalsmu[0]:

        mu_cs0 = np.zeros(q) 

        mu_cs0_all[catvaluetoindexmu[j], :] = mu_cs0


else:
    for j in allcatvalsmu[0]:

        mu_cs0 = np.zeros(q) 
        mu_cs0 = np.array(mu_cs0)[0, :]

        mu_cs0_all[catvaluetoindexmu[j], :] = mu_cs0


print('check mu_cs0_all')
print(mu_cs0_all)

piparamprior=1/K 

for g in allcatvalsmu[0]:
    piprob[0, :] = rng.dirichlet(alpha=np.ones(K)*piparamprior, size=1)  


if mudependentong==0:
    for k in range(0,K):
        for g in allcatvalsmu[0]:
            Sigma[0, k, catvaluetoindexSigma[j], :] = invwishart.rvs(df=nu, scale=Phi, size=1, random_state=rng) 
            print('j is ' + str(j) + ' k is ' + str(k))

            mu_cs0 = mu_cs0_all[catvaluetoindexmu[j], :] 

            print('check mu_cs0')
            print(mu_cs0)
            print(np.shape(mu_cs0))
            print(np.abs(mu_cs0))
            print('check Sigma_cs0')
            print(np.diag(np.abs(mu_cs0)))
            mu[0, k, catvaluetoindexmu[j], :] = rng.multivariate_normal(mean=mu_cs0,cov=Sigma[0, k, catvaluetoindexSigma[j], :]/(beta_kg_prior),size=1)  


else:
    for k in range(K):
        for j in allcatvalsmu[0]:
            Sigma[0, k, catvaluetoindexSigma[j], :] = invwishart.rvs(df=nu, scale=Phi, size=1, random_state=rng) 
            print('j is ' + str(j) + ' k is ' + str(k))

            mu_cs0 = mu_cs0_all[catvaluetoindexmu[j], :] 
            mu_cs0 = np.array(mu_cs0)[0, :]

            print('check mu_cs0')
            print(mu_cs0)
            print(np.shape(mu_cs0))
            print(np.abs(mu_cs0))
            print('check Sigma_cs0')
            print(np.diag(np.abs(mu_cs0)))
            mu[0, k, catvaluetoindexmu[j], :] = rng.multivariate_normal(mean=mu_cs0,cov=Sigma[0, k, catvaluetoindexSigma[j], :]/(beta_kg_prior),size=1)  


for k in range(0,K):
    for j in range(0,p):

        Psi[0, k, j, 0:categorical_columns_dims[j]] = rng.dirichlet(alpha=np.ones(categorical_columns_dims[j]) / categorical_columns_dims[j], size=1)


print('check piprob')
print(piprob[0, :])
print('check Psi')
print(Psi[0, :])
print('check mu')
print(mu[0, :])
print('check Sigma')
print(Sigma[0, :])

import pickle
import time
start_time = time.time()

def computejoint(z,mu,Sigma,Psi,pi):

    zint = z.astype(int)

    gvec = np.empty(N)
    for k in range(0,K):
        idx = np.where(zint == k)[0]
        if idx.size:
            gvec[idx] = multivariate_normal.logpdf(x=continuousdata[idx], mean=mu[k,:], cov=Sigma[k,:])


    xllh = gvec.sum()
    muprior=0
    Sigmaprior=0
    for k in range(0,K):
        muprior=muprior+multivariate_normal.logpdf(x=mu[k,:],mean=mu_cs0_all[0,:],cov=Sigma[k,:]/beta_kg_prior)
        Sigmaprior=Sigmaprior+invwishart.logpdf(x=Sigma[k,:], df=nu, scale=Phi)


    cllh=0
    for j in range(0,p):

        cllh=cllh+np.log(Psi[zint, j, catcodes[:, j]]).sum()


    psiprior=0
    for k in range(0,K):
        for j in range(0,p):
            psiprior=psiprior+dirichlet.logpdf(x=Psi[k,j,:],alpha=np.ones(categorical_columns_dims[j]) / categorical_columns_dims[j])


    zllh=np.log(pi[zint]).sum()
    piprior=dirichlet.logpdf(x=pi, alpha=np.ones(K)*piparamprior)
    return xllh+muprior+Sigmaprior+cllh+psiprior+zllh+piprior


timerecord=np.zeros(iterations)
jointdistval=np.zeros(iterations)

catcodes = fulldataframe[modelled_categorical_columns].to_numpy().astype(int)  

gibbs_compute_elapsed = 0.0

for iteration in range(1, iterations):
    print('we are at iteration')
    print(iteration)

    iter_t0 = time.perf_counter()   

    for clusterno in range(0,K):
        for j in range(0,p):
                alphatemp = np.zeros(categorical_columns_dims[j])
                for g_j in range(0,categorical_columns_dims[j]):

                    count_kgj = (z[:, 0] == clusterno) & (fulldataframe[modelled_categorical_columns[j]] == g_j)
                    alphatemp[g_j] = 1 / categorical_columns_dims[j] + np.sum(count_kgj)


                Psi[iteration, clusterno, j, 0:categorical_columns_dims[j]] = rng.dirichlet(alpha=alphatemp, size=1)


    for clusterno in range(K):

        Phi_new = Phi

        for catvalue in allcatvalsSigma[0]:
            count_kc = (z[:, 0] == clusterno) & (combinedcategoriesSigma == catvalue)
            locations_kc = np.where((z[:, 0] == clusterno) & (combinedcategoriesSigma == catvalue) )
            n_kc = np.sum(count_kc)

            if n_kc > 0:  
                Phivec2= np.mean(continuousdata[locations_kc], axis=0) - mu_cs0_all[catvaluetoindexmu[catvalue], :]

                Phivec1 = continuousdata[locations_kc] - np.mean(continuousdata[locations_kc], axis=0)  
                Phi_new = Phi_new + Phivec1.T @ Phivec1                                                      
                Phi_new=Phi_new+(beta_kg_prior*n_kc/(beta_kg_prior+n_kc))*np.outer(Phivec2,Phivec2.T)

            Sigma[iteration, clusterno, catvaluetoindexSigma[catvalue], :] = invwishart.rvs(df=nu + n_kc, scale=Phi_new, size=1, random_state=rng)  


    for clusterno in range(K):
        for catvalue in allcatvalsmu[0]:
            count_kc = (z[:, 0] == clusterno) & (combinedcategoriesmu == catvalue) 
            locations_kc = np.where((z[:, 0] == clusterno) & (combinedcategoriesmu == catvalue) )
            n_kc = np.sum(count_kc)

            mu_cs0temp = mu_cs0_all[catvaluetoindexmu[catvalue], :]
            Sigma_cs0temp = Sigma[iteration, clusterno, catvaluetoindexmu[g], :]/beta_kg_prior
            if n_kc == 0:  

                mu_cs0new = mu_cs0temp
                Sigma_cs0new = Sigma_cs0temp
            else:

                mu_cs0new=(beta_kg_prior*mu_cs0_all[catvaluetoindexmu[catvalue], :]+np.sum(continuousdata[locations_kc[0],:], axis=0) )/(beta_kg_prior+n_kc)
                Sigma_cs0new = Sigma[iteration, clusterno, catvaluetoindexmu[g], :]/(beta_kg_prior+n_kc)

            mu[iteration, clusterno, catvaluetoindexmu[catvalue], :] = multivariate_normal.rvs(mean=mu_cs0new, cov=Sigma_cs0new, size=1, random_state=rng)  


    alphatempPi = np.zeros(K)
    for clusterno in range(K):
        count_k = (z[:, 0] == clusterno) 
        n_k = np.sum(count_k)
        alphatempPi[clusterno] = piparamprior + n_k


    piprob[iteration, :] = rng.dirichlet(alpha=alphatempPi, size=1)

    logpi = np.log(piprob[iteration, :])                       
    zlogprobmatrix = np.empty((N, K))
    for clusterno in range(K):
        zlogprobmatrix[:, clusterno] = multivariate_normal.logpdf(
            continuousdata,
            mean=mu[iteration, clusterno, 0, :],
            cov=Sigma[iteration, clusterno, 0, :])


    for j in range(0, p):
        zlogprobmatrix += np.log(Psi[iteration, :, j, :])[:, catcodes[:, j]].T   


    zlogprobmatrix += logpi[None, :]

    zexpprobmatrix = np.exp(zlogprobmatrix)
    rowsum = zexpprobmatrix.sum(axis=1)
    underflow = rowsum == 0                                     
    if np.any(underflow):
        zlogprobmatrix[underflow] -= zlogprobmatrix[underflow].max(axis=1, keepdims=True)
        zexpprobmatrix[underflow] = np.exp(zlogprobmatrix[underflow])
        rowsum[underflow] = zexpprobmatrix[underflow].sum(axis=1)
    zprobnormalised[:, :] = zexpprobmatrix / rowsum[:, None]

    zuniforms = rng.random(N)
    zcdf = np.cumsum(zprobnormalised, axis=1)
    zcdf /= zcdf[:, -1][:, None]
    z[:, 0] = (zcdf <= zuniforms[:, None]).sum(axis=1)

    gibbs_compute_elapsed += time.perf_counter() - iter_t0     
    timerecord[iteration] = gibbs_compute_elapsed              

    z_history[iteration] = z[:, 0]

    if iteration % 10 == 1:
        ztemp = z.astype(int)
        print('predicted hard clusters')
        print(Counter(ztemp[:,0]))

        print('true clusters')
        print(Counter(fulldataframe['clusterlabel']))

    outputmu = open('mu'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
    pickle.dump(mu, outputmu)
    outputmu.close()

    outputzprob = open('zprob'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
    pickle.dump(zprobnormalised, outputzprob)
    outputzprob.close()

    outputz = open('z'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
    pickle.dump(z, outputz)
    outputz.close()

    outputPsi = open('Psi'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
    pickle.dump(Psi, outputPsi)
    outputPsi.close()

    outputSigma = open('Sigma'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
    pickle.dump(Sigma, outputSigma)
    outputSigma.close()

    outputpiprob = open('piprob'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
    pickle.dump(piprob, outputpiprob)
    outputpiprob.close()

    outputiteration = open('iteration'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
    pickle.dump(iteration, outputiteration)
    outputiteration.close()

    outputtimerecord = open('timerecord'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
    pickle.dump(timerecord, outputtimerecord)
    outputtimerecord.close()

    outputjointdistval = open('jointdistval'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
    pickle.dump(jointdistval, outputjointdistval)
    outputjointdistval.close()

    jointdistval[iteration]=computejoint(z=z[:, 0],mu=mu[iteration,:,0,:],Sigma=Sigma[iteration,:,0,:],Psi=Psi[iteration,:,:,:],pi=piprob[iteration,:])
    print('joint density val')
    print(jointdistval[iteration])    


end_time = time.time()
elapsed_time = end_time - start_time

print('elapsed_time')
print(elapsed_time)

COND_THIN = 1                                   
cond_iters = np.arange(1, iterations, COND_THIN)
n_cond = len(cond_iters)

cond_mu_mean   = np.zeros((n_cond, K, q))
cond_beta_eff  = np.zeros((n_cond, K))
cond_sig_df    = np.zeros((n_cond, K))
cond_sig_scale = np.zeros((n_cond, K, q, q))
cond_psi_alpha = np.zeros((n_cond, K, p, int(np.max(categorical_columns_dims))))
cond_pi_alpha  = np.zeros((n_cond, K))          

for si, it in enumerate(cond_iters):
    z_prev = z_history[it - 1]                  
    for clusterno in range(0, K):
        idx = np.where(z_prev == clusterno)[0]
        n_kc = len(idx)

        for j in range(0, p):
            alphatemp = np.zeros(categorical_columns_dims[j])
            for g_j in range(0, categorical_columns_dims[j]):
                count_kgj = (z_prev == clusterno) & (catcodes[:, j] == g_j)
                alphatemp[g_j] = 1 / categorical_columns_dims[j] + np.sum(count_kgj)


            cond_psi_alpha[si, clusterno, j, 0:categorical_columns_dims[j]] = alphatemp


        Phi_new = Phi
        if n_kc > 0:
            Phivec1 = continuousdata[idx] - np.mean(continuousdata[idx], axis=0)         
            Phivec2 = np.mean(continuousdata[idx], axis=0) - mu_cs0_all[0, :]       
            Phi_new = Phi_new + Phivec1.T @ Phivec1                                           
            Phi_new = Phi_new + (beta_kg_prior * n_kc / (beta_kg_prior + n_kc)) * np.outer(Phivec2, Phivec2.T)
        cond_sig_df[si, clusterno]    = nu + n_kc
        cond_sig_scale[si, clusterno] = Phi_new

        cond_mu_mean[si, clusterno] = (beta_kg_prior * mu_cs0_all[0, :]
                                       + (continuousdata[idx].sum(axis=0) if n_kc else 0)) \
                                      / (beta_kg_prior + n_kc)
        cond_beta_eff[si, clusterno] = beta_kg_prior + n_kc

        cond_pi_alpha[si, clusterno] = piparamprior + n_kc          


print('conditional parameters reconstructed for', n_cond, 'iterations (untimed, post-run)')

end_time2 = time.time()
elapsed_time2 = end_time2 - end_time

print('elapsed_time2')
print(elapsed_time2)

import numpy as np
from scipy.stats import rankdata
from scipy.special import ndtri

def _normalized_acf(x):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[:n].real
    if acf[0] <= 0:
        return None
    return acf / acf[0]


def integrated_autocorr_time(x, method='geyer', c=5.0):
    rho = _normalized_acf(x)
    if rho is None:
        return 1.0
    n = len(rho)

    if method == 'geyer':
        m_max = (n - 1) // 2
        if m_max < 1:
            return 1.0
        gamma = rho[0:2 * m_max:2] + rho[1:2 * m_max + 1:2]   
        nonpos = np.nonzero(gamma <= 0)[0]                    
        M = nonpos[0] if nonpos.size else len(gamma)
        gamma = gamma[:M]
        if gamma.size == 0:
            return 1.0
        gamma = np.minimum.accumulate(gamma)                  
        return max(float(-1.0 + 2.0 * gamma.sum()), 1.0)

    elif method == 'sokal':
        taus = 2.0 * np.cumsum(rho) - 1.0
        win = np.arange(n) < c * taus
        M = np.argmin(win) if np.any(~win) else n - 1
        return max(float(taus[M]), 1.0)

    raise ValueError("method must be 'geyer' or 'sokal'")


def rank_normalize(x):
    r = rankdata(x)
    return ndtri((r - 0.375) / (len(x) + 0.25))


def geweke_z(x, first=0.1, last=0.5, normalize=True, method='geyer'):
    x = np.asarray(x, dtype=float)
    if normalize:
        x = rank_normalize(x)
    n = len(x)
    a, b = x[:int(first * n)], x[-int(last * n):]
    tau = integrated_autocorr_time(x, method=method)   
    va = np.var(a, ddof=1) * tau / len(a)
    vb = np.var(b, ddof=1) * tau / len(b)
    return (a.mean() - b.mean()) / np.sqrt(va + vb)


def geweke_zscores(x, n_candidates=20, **kw):
    n = len(x)
    cands = np.unique(np.linspace(0, n // 2, n_candidates).astype(int))
    return np.array([[b, geweke_z(x[b:], **kw)] for b in cands])


def geweke_select_burnin(x, n_candidates=20, zthresh=2.0, consec=2, min_len=200, **kw):
    n = len(x)
    cands = np.unique(np.linspace(0, n // 2, n_candidates).astype(int))
    zs = np.array([[b, geweke_z(x[b:], **kw)] for b in cands if n - b >= min_len])
    if len(zs) < consec:
        return None, zs
    passing = np.abs(zs[:, 1]) < zthresh
    for i in range(len(zs) - consec + 1):
        if passing[i:i + consec].all():
            return int(zs[i, 0]), zs


    return None, zs


def phase1_geweke(run_block, initial_block=1500, block=1000, max_total=5000, **kw):
    lj = run_block(initial_block)
    while True:
        B, zs = geweke_select_burnin(np.asarray(lj), **kw)
        if B is not None:
            return B, lj, zs
        if len(lj) >= max_total:
            return None, lj, zs
        lj = run_block(min(block, max_total - len(lj)))


B, zs = geweke_select_burnin(jointdistval[1:])

print(B)

from scipy.optimize import linear_sum_assignment

def error_permuted(truevals, estimatedvals):

    distance_matrix = np.zeros((Ktrue, K))

    for i in range(0,Ktrue):
        for j in range(0,K):

            distance_matrix[i, j] = np.mean( np.abs(truevals[i,:] - estimatedvals[j,:]) )


    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    matched_true_means = truevals[row_ind,:]
    matched_estimated_means = estimatedvals[col_ind,:]

    mean_error = np.mean( np.abs(matched_true_means - matched_estimated_means) )

    print(f"Matched True Means:\n{matched_true_means}")
    print(f"Matched Estimated Means:\n{matched_estimated_means}")

    print(f"Mean Error: {mean_error:.10f}")

    print(f"true_ind: {row_ind}")
    print(f"estimated_ind: {col_ind}")

    return mean_error, row_ind, col_ind, matched_true_means, matched_estimated_means


def error_nopermute(truevals, estimatedvals, row_ind, col_ind):

    matched_true_means = truevals[row_ind,:]
    matched_estimated_means = estimatedvals[col_ind,:]

    mean_error = np.mean( np.abs(matched_true_means - matched_estimated_means) )

    print(f"Matched True Means:\n{matched_true_means}")
    print(f"Matched Estimated Means:\n{matched_estimated_means}")

    print(f"Mean Error: {mean_error:.10f}")

    print(f"true_ind: {row_ind}")
    print(f"estimated_ind: {col_ind}")

    return mean_error, row_ind, col_ind, matched_true_means, matched_estimated_means


def errors_from_means(mu_pm, Sigma_pm, Psi_pm, pi_pm,
                      truemu_scaled, trueSigma_scaled, truepsi_arr, trueclusterprob_arr):
    D = np.zeros((Ktrue, K))
    for a in range(0, Ktrue):
        for b in range(0, K):
            D[a, b] = np.mean(np.abs(truemu_scaled[a] - mu_pm[b]))


    row_ind, col_ind = linear_sum_assignment(D)
    mu_err = np.mean(np.abs(truemu_scaled[row_ind] - mu_pm[col_ind]))
    sigma_err = np.mean(np.abs(trueSigma_scaled[row_ind] - Sigma_pm[col_ind]))
    psi_errs = np.zeros(p)
    for j in range(0, p):
        s = np.sum(Psi_pm[col_ind, j, :], axis=1)
        psi_errs[j] = np.mean(np.abs(truepsi_arr[row_ind, j, :] - Psi_pm[col_ind, j, :] / s[:, np.newaxis]))


    psi_err = np.mean(psi_errs)
    pi_err = np.mean(np.abs(trueclusterprob_arr[row_ind] - pi_pm[col_ind]))
    return mu_err, sigma_err, psi_err, pi_err, row_ind, col_ind


def first_crossing_time(error_trace, time_trace, target_error):
    valid = ~np.isnan(error_trace)
    hits = np.where(valid & (error_trace <= target_error))[0]
    if hits.size == 0:
        return np.nan, -1
    t = int(hits[0])
    return time_trace[t], t


if B is None:                                                                                                      
    B = 50  
burnin=max(B, 50)   
burn_in=max(B, 50)
lastiter=iterations

muposteriormean=np.mean(mu[burnin:lastiter,:,0,:], axis=0) 
Sigmaposteriormean=np.mean(Sigma[burnin:lastiter,:,0,:], axis=0)  

print(np.shape(muposteriormean))
print(np.shape(Sigmaposteriormean))

print(muposteriormean)

print('mean in the original scale')

error_permuted(truemu_kg[:,0,:], scaler.inverse_transform(muposteriormean) )

print('mean in the standardised scale')

mu_error, mu_row_ind, mu_col_ind, matched_true_mu, matched_estimated_mu=error_permuted(scaler.transform(truemu_kg[:,0,:]), muposteriormean)

print('covariance in the original scale')
estimatedcovogscale=np.array([ np.diag(scaler.scale_) @ Sigmaposteriormean[k,:] @ np.diag(scaler.scale_) for k in range(K)])

error_permuted(trueSigma_k[:,0,:], estimatedcovogscale )

print('covariance in the standardised scale')
trueSigma_kscaled=np.array([ np.diag(1/scaler.scale_) @ trueSigma_k[k, catvaluetoindexmu[g], :] @ np.diag(1/scaler.scale_) for k in range(Ktrue)])

sigma_error, sigma_row_ind, sigma_col_ind, matched_true_sigma, matched_estimated_sigma=error_nopermute(trueSigma_kscaled, Sigmaposteriormean, mu_row_ind, mu_col_ind) 

error_every = 1   

truemu_scaled_trace    = scaler.transform(truemu_kg[:, 0, :])
trueSigma_scaled_trace = np.array([np.diag(1/scaler.scale_) @ trueSigma_k[a, 0, :] @ np.diag(1/scaler.scale_)
                                   for a in range(Ktrue)])

mu_error_trace    = np.full(iterations, np.nan)
sigma_error_trace = np.full(iterations, np.nan)
psi_error_trace   = np.full(iterations, np.nan)
pi_error_trace    = np.full(iterations, np.nan)

_mu_sum    = np.zeros((K, q))
_Sigma_sum = np.zeros((K, q, q))
_Psi_sum   = np.zeros((K, p, int(np.max(categorical_columns_dims))))
_pi_sum    = np.zeros(K)
_cnt = 0
for t in range(burn_in, iterations):
    _mu_sum    += mu[t, :, 0, :]
    _Sigma_sum += Sigma[t, :, 0, :]
    _Psi_sum   += Psi[t, :, :, :]
    _pi_sum    += piprob[t, :]
    _cnt += 1
    if (t % error_every == 0) or (t == iterations - 1):
        me, se, pse, pie, _, _ = errors_from_means(
            _mu_sum/_cnt, _Sigma_sum/_cnt, _Psi_sum/_cnt, _pi_sum/_cnt,
            truemu_scaled_trace, trueSigma_scaled_trace, truepsi, trueclusterprob)
        mu_error_trace[t]    = me
        sigma_error_trace[t] = se
        psi_error_trace[t]   = pse
        pi_error_trace[t]    = pie


print('error-vs-time trace: final (mu, Sigma, Psi, pi) errors:')
print(mu_error_trace[iterations-1], sigma_error_trace[iterations-1],
      psi_error_trace[iterations-1], pi_error_trace[iterations-1])

for _tracename, _tracearr in [('mu_error_trace', mu_error_trace),
                              ('sigma_error_trace', sigma_error_trace),
                              ('psi_error_trace', psi_error_trace),
                              ('pi_error_trace', pi_error_trace),
                              ('timerecord', timerecord)]:
    _tracef = open(_tracename+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
    pickle.dump(_tracearr, _tracef)
    _tracef.close()


def most_frequent_per_column(arr):
    n_rows, n_cols = arr.shape
    most_frequent = np.zeros(n_cols, dtype=int)  

    for col in range(n_cols):
        col_values = arr[:, col]
        unique_values, counts = np.unique(col_values, return_counts=True)
        most_frequent[col] = unique_values[np.argmax(counts)]  


    return most_frequent


from scipy.stats import mode

mapping = dict(zip(mu_col_ind, mu_row_ind))

predictedclusters=z[:,0] 

orderedclusters=np.vectorize(lambda x: mapping.get(x, -1))(predictedclusters)

print(np.sum(orderedclusters == component)/len(orderedclusters) )

hardclustermatchesprob=np.sum(orderedclusters == component)/len(orderedclusters)

predpsiprob=np.zeros((Ktrue,p,2))

Psiposteriormean=np.mean(Psi[burnin:lastiter,:,:,:], axis=0) 

Psi_error=np.zeros(p)

for j in range(0,p):
    print('we are at column '+str(j))

    Psiparamsum=np.sum(Psiposteriormean[mu_col_ind,j,:], axis=1)
    print('truepsi')
    print( truepsi[:,j,:] )
    print('rearranged Psiparam prob')
    print( Psiposteriormean[mu_col_ind,j,:]/Psiparamsum[:,np.newaxis] )

    print('error')
    Psi_error[j]=np.mean( np.abs(truepsi[:,j,:]-Psiposteriormean[mu_col_ind,j,:]/Psiparamsum[:,np.newaxis]) )

    print(Psi_error[j])


print('Psi error averaged over p columns')
Psi_error_mean=np.mean(Psi_error)
print(Psi_error_mean)

piposteriormean=np.mean(piprob[burnin:lastiter,mu_col_ind], axis=0)

estimateclusterprob=piposteriormean 

print(piposteriormean)
print(trueclusterprob)
print(np.abs(trueclusterprob-estimateclusterprob))

pi_error=np.mean(np.abs(trueclusterprob-estimateclusterprob)) 

print(pi_error)

mu_error_vec=mu_error
mu_row_ind_vec=mu_row_ind
mu_col_ind_vec=mu_col_ind

sigma_error_vec=sigma_error
sigma_row_ind_vec=sigma_row_ind
sigma_col_ind_vec=sigma_col_ind

outputmu_error_vec=open('mu_error_vec'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(mu_error_vec, outputmu_error_vec)
outputmu_error_vec.close()

outputmu_row_ind_vec=open('mu_row_ind_vec'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(mu_row_ind_vec, outputmu_row_ind_vec)
outputmu_row_ind_vec.close()

outputmu_col_ind_vec=open('mu_col_ind_vec'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(mu_col_ind_vec, outputmu_col_ind_vec)
outputmu_col_ind_vec.close()

outputsigma_error_vec=open('sigma_error_vec'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(sigma_error_vec, outputsigma_error_vec)
outputsigma_error_vec.close()

outputsigma_row_ind_vec=open('sigma_row_ind_vec'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(sigma_row_ind_vec, outputsigma_row_ind_vec)
outputsigma_row_ind_vec.close()

outputsigma_col_ind_vec=open('sigma_col_ind_vec'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(sigma_col_ind_vec, outputsigma_col_ind_vec)
outputsigma_col_ind_vec.close()

outputhardclustermatchesprob=open('hardclustermatchesprob'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(hardclustermatchesprob, outputhardclustermatchesprob)
outputhardclustermatchesprob.close()

outputPsi_error_mean=open('Psi_error_mean'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(Psi_error_mean, outputPsi_error_mean)
outputPsi_error_mean.close()

outputpi_error=open('pi_error'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(pi_error, outputpi_error)
outputpi_error.close()

outputelapsed_time = open('elapsed_time'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(elapsed_time, outputelapsed_time)
outputelapsed_time.close()  

d=d_1

logposteriortrue=0.0

trueclusterprobfull=np.zeros(K)
trueclusterprobfull[mu_col_ind]=trueclusterprob

truemu_kgfull=np.zeros((K,q))
truemu_kgfull[mu_col_ind,:]=scaler.transform(truemu_kg[:,0,:])

trueSigma_full=np.zeros((K,q,q))
trueSigma_full[mu_col_ind,:]=np.array([ np.diag(1/scaler.scale_) @ trueSigma_k[k, 0, :] @ np.diag(1/scaler.scale_) for k in range(Ktrue)]) 

truepsi_full=np.zeros((K,p,d))
for j in range(0,p):
    truepsi_full[mu_col_ind,j,:]=truepsi[:,j,:]


logposteriortrueall=computejoint(z=component,mu=truemu_kgfull,Sigma=trueSigma_full,Psi=truepsi_full,pi=trueclusterprobfull)
logposteriorall=np.percentile(jointdistval[burn_in:lastiter], 5)

print(logposteriortrueall)
print(logposteriorall)

coverageresults=logposteriortrueall>logposteriorall

print(coverageresults)

outputlogposteriorall=open('logposteriorall'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(logposteriorall, outputlogposteriorall)
outputlogposteriorall.close()

outputlogposteriortrueall=open('logposteriortrueall'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(logposteriortrueall, outputlogposteriortrueall)
outputlogposteriortrueall.close()

outputcoverageresults=open('coverageresults'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(coverageresults, outputcoverageresults)
outputcoverageresults.close()

outputburnin=open('burnin'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(burnin, outputburnin)
outputburnin.close()

print('done all')

print(mu_error_vec)

print(sigma_error_vec)

print(hardclustermatchesprob)

print(Psi_error_mean)

print(pi_error)

import numpy as np
import pickle
from scipy.special import logsumexp, multigammaln
from scipy.linalg import solve_triangular, cholesky
from scipy.optimize import linear_sum_assignment

FLO = 0.05           

def _mvn_logpdf_batch(Xe, mean, cov):
    qd = len(mean)
    L = cholesky(cov, lower=True)
    sol = solve_triangular(L, (Xe - mean).T, lower=True)
    return (-0.5 * np.sum(sol ** 2, axis=0)
            - np.sum(np.log(np.diag(L)))
            - 0.5 * qd * np.log(2 * np.pi))


def _iw_logpdf_batch(Se_inv, Se_logdet, df, scale):
    qd = scale.shape[0]
    Ls = cholesky(scale, lower=True)
    logdet_scale = 2.0 * np.sum(np.log(np.diag(Ls)))
    tr = np.einsum('ij,mji->m', scale, Se_inv)
    return (0.5 * df * logdet_scale
            - 0.5 * df * qd * np.log(2.0)
            - multigammaln(0.5 * df, qd)
            - 0.5 * (df + qd + 1.0) * Se_logdet
            - 0.5 * tr)


def _dirichlet_logpdf_batch(Xe, alpha):
    from scipy.special import gammaln, xlogy
    Xn = Xe / Xe.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        return (gammaln(alpha.sum()) - gammaln(alpha).sum()
                + np.sum(xlogy(alpha - 1.0, Xn), axis=1))


def _rowshift_cumsum(L):
    c = L.max(axis=1)
    return c, np.cumsum(np.exp(L - c[:, None]), axis=1)


def _coverage_from_cumsum(c_eval, Cum_eval, c_truth, Cum_truth, n_grid, m_grid, flo=FLO):
    out = np.zeros(len(n_grid), dtype=bool)
    ld_true_grid = np.full(len(n_grid), np.nan)          
    thr_grid = np.full(len(n_grid), np.nan)              
    ld_eval_last = np.empty(0)                           
    for gi, (nT, mT) in enumerate(zip(n_grid, m_grid)):
        if nT < 2 or mT < 2:
            continue
        ld_eval = c_eval[:mT] + np.log(Cum_eval[:mT, nT - 1]) - np.log(nT)
        ld_true = c_truth + np.log(Cum_truth[nT - 1]) - np.log(nT)
        thr = np.quantile(ld_eval, flo)                  
        out[gi] = ld_true >= thr
        ld_true_grid[gi] = ld_true                       
        thr_grid[gi] = thr                               
        ld_eval_last = ld_eval                           


    return out, ld_true_grid, thr_grid, ld_eval_last     


def _iat_geyer(z):                                                                
    z = np.asarray(z, float); n = len(z)                                          
    if n < 8:                                                                     
        return 1.0                                                                
    z = z - z.mean()                                                              
    if np.dot(z, z) == 0:                                                         
        return 1.0                                                                
    f = np.fft.fft(z, n=2 * n)                                                     
    acov = np.fft.ifft(f * np.conjugate(f))[:n].real / n                          
    rho = acov / acov[0]                                                          
    t = 1; s = 0.0                                                                
    while t + 1 < n:                                                              
        g = rho[t] + rho[t + 1]                                                   
        if g < 0:                                                                 
            break                                                                 
        s += g; t += 2                                                            
    return max(1.0 + 2.0 * s, 1.0)                                                


def _mcse_logmean_trace(Lt, n_grid):                                              
    out = np.full(len(n_grid), np.nan)                                            
    for gi, nT in enumerate(n_grid):                                              
        if nT < 8:                                                                
            continue                                                              
        lg = Lt[:nT]                                                              
        c = lg.max()                                                              
        g = np.exp(lg - c)               
        mean_g = g.mean()                                                         
        if mean_g <= 0:                                                           
            continue                                                              
        sig2 = g.var(ddof=1)                                                      
        tau = _iat_geyer(g)                                                       
        out[gi] = np.sqrt(sig2 * tau / nT) / mean_g   


    return out                                                                    


def block_coverage_trace(cond, chains, truth, T_grid_pos, eval_pos,
                         level=0.95, cat_dims=None, verbose=False):
    S, K = cond['beta_eff'].shape
    p, d = cond['psi_alpha'].shape[2], cond['psi_alpha'].shape[3]
    if cat_dims is None:
        cat_dims = [d] * p
    T_grid_pos = np.asarray(T_grid_pos); eval_pos = np.asarray(eval_pos)
    n_grid = T_grid_pos
    m_grid = np.searchsorted(eval_pos, T_grid_pos, side='left')

    ng = len(T_grid_pos)
    res = {'mu': np.zeros((K, ng), bool),
           'Sigma': np.zeros((K, ng), bool),
           'Psi': np.zeros((K, p, ng), bool),
           'pi': np.zeros(ng, bool),                       

           'mu_ld_true': np.full((K, ng), np.nan), 'mu_thr': np.full((K, ng), np.nan),        
           'Sigma_ld_true': np.full((K, ng), np.nan), 'Sigma_thr': np.full((K, ng), np.nan),  
           'Psi_ld_true': np.full((K, p, ng), np.nan), 'Psi_thr': np.full((K, p, ng), np.nan),
           'pi_ld_true': np.full(ng, np.nan), 'pi_thr': np.full(ng, np.nan),                  
           'mu_ld_eval_last': {}, 'Sigma_ld_eval_last': {}, 'Psi_ld_eval_last': {},          

           'mu_ld_true_mcse': np.full((K, ng), np.nan),                                  
           'Sigma_ld_true_mcse': np.full((K, ng), np.nan),                               
           'Psi_ld_true_mcse': np.full((K, p, ng), np.nan),                              
           'pi_ld_true_mcse': np.full(ng, np.nan)}                                       

    for k in range(K):

        Me = chains['mu'][eval_pos, k, :]
        L = np.empty((len(eval_pos), S)); Lt = np.empty(S)
        for s in range(S):
            cov_s = chains['Sigma'][s, k] / cond['beta_eff'][s, k]
            L[:, s] = _mvn_logpdf_batch(Me, cond['mu_mean'][s, k], cov_s)
            Lt[s] = _mvn_logpdf_batch(truth['mu'][k][None, :], cond['mu_mean'][s, k], cov_s)[0]


        ce, Ce = _rowshift_cumsum(L)
        ct, Ct = _rowshift_cumsum(Lt[None, :])
        res['mu'][k], res['mu_ld_true'][k], res['mu_thr'][k], res['mu_ld_eval_last'][k] = \
            _coverage_from_cumsum(ce, Ce, ct[0], Ct[0], n_grid, m_grid, flo=1.0-level)   
        res['mu_ld_true_mcse'][k] = _mcse_logmean_trace(Lt, n_grid)                       

        Se = chains['Sigma'][eval_pos, k]
        inv_e = np.linalg.inv(Se); _, ld_e = np.linalg.slogdet(Se)
        inv_t = np.linalg.inv(truth['Sigma'][k])[None, ...]
        _, ld_t = np.linalg.slogdet(truth['Sigma'][k][None, ...])
        L = np.empty((len(eval_pos), S)); Lt = np.empty(S)
        for s in range(S):
            L[:, s] = _iw_logpdf_batch(inv_e, ld_e, cond['sig_df'][s, k], cond['sig_scale'][s, k])
            Lt[s] = _iw_logpdf_batch(inv_t, ld_t, cond['sig_df'][s, k], cond['sig_scale'][s, k])[0]


        ce, Ce = _rowshift_cumsum(L)
        ct, Ct = _rowshift_cumsum(Lt[None, :])
        res['Sigma'][k], res['Sigma_ld_true'][k], res['Sigma_thr'][k], res['Sigma_ld_eval_last'][k] = \
            _coverage_from_cumsum(ce, Ce, ct[0], Ct[0], n_grid, m_grid, flo=1.0-level)   
        res['Sigma_ld_true_mcse'][k] = _mcse_logmean_trace(Lt, n_grid)                    

        for j in range(p):
            dj = cat_dims[j]
            Pe = chains['Psi'][eval_pos, k, j, :dj]
            L = np.empty((len(eval_pos), S)); Lt = np.empty(S)
            for s in range(S):
                a = cond['psi_alpha'][s, k, j, :dj]
                L[:, s] = _dirichlet_logpdf_batch(Pe, a)
                Lt[s] = _dirichlet_logpdf_batch(truth['Psi'][k, j, :dj][None, :], a)[0]


            ce, Ce = _rowshift_cumsum(L)
            ct, Ct = _rowshift_cumsum(Lt[None, :])
            res['Psi'][k, j], res['Psi_ld_true'][k, j], res['Psi_thr'][k, j], _pld = \
                _coverage_from_cumsum(ce, Ce, ct[0], Ct[0], n_grid, m_grid, flo=1.0-level)   
            res['Psi_ld_eval_last'][(k, j)] = _pld                                            
            res['Psi_ld_true_mcse'][k, j] = _mcse_logmean_trace(Lt, n_grid)                   


        if verbose:
            print(f'  cluster {k} done')


    Pe = chains['pi'][eval_pos, :]                          
    L = np.empty((len(eval_pos), S)); Lt = np.empty(S)      
    for s in range(S):                                     
        a = cond['pi_alpha'][s, :]                         
        L[:, s] = _dirichlet_logpdf_batch(Pe, a)          
        Lt[s] = _dirichlet_logpdf_batch(truth['pi'][None, :], a)[0]   


    ce, Ce = _rowshift_cumsum(L)                           
    ct, Ct = _rowshift_cumsum(Lt[None, :])                 
    res['pi'], res['pi_ld_true'], res['pi_thr'], res['pi_ld_eval_last'] = \
        _coverage_from_cumsum(ce, Ce, ct[0], Ct[0], n_grid, m_grid, flo=1.0-level)   
    res['pi_ld_true_mcse'] = _mcse_logmean_trace(Lt, n_grid)                          

    return res


def prepare_window(B, mu, Sigma, Psi, piprob, cond_iters,               
                   cond_mu_mean, cond_beta_eff, cond_sig_df, cond_sig_scale,
                   cond_psi_alpha, cond_pi_alpha,                        
                   true_mu, true_Sigma, true_Psi, true_pi, T_max=None):  
    T_max = cond_iters.max() if T_max is None else T_max
    mask = (cond_iters >= B) & (cond_iters <= T_max)
    sel = cond_iters[mask]
    chains = {'mu': mu[sel, :, 0, :], 'Sigma': Sigma[sel, :, 0, :], 'Psi': Psi[sel],
              'pi': piprob[sel]}                                          
    cond = {'mu_mean': cond_mu_mean[mask], 'beta_eff': cond_beta_eff[mask],
            'sig_df': cond_sig_df[mask], 'sig_scale': cond_sig_scale[mask],
            'psi_alpha': cond_psi_alpha[mask],
            'pi_alpha': cond_pi_alpha[mask]}                             
    pmm = chains['mu'].mean(axis=0)
    D = np.mean(np.abs(true_mu[:, None, :] - pmm[None, :, :]), axis=2)
    r, c = linear_sum_assignment(D)
    perm = np.empty(true_mu.shape[0], dtype=int); perm[r] = c

    chains = {kk: v[:, perm] for kk, v in chains.items()}
    cond = {kk: v[:, perm] for kk, v in cond.items()}
    truth = {'mu': true_mu, 'Sigma': true_Sigma, 'Psi': true_Psi, 'pi': true_pi}    
    return cond, chains, truth, sel


start_timecov=time.time()

B_cov = burn_in                      

_cond, _chains, _truth, _sel = prepare_window(
    B_cov, mu, Sigma, Psi, piprob, cond_iters,                     
    cond_mu_mean, cond_beta_eff, cond_sig_df, cond_sig_scale, cond_psi_alpha, cond_pi_alpha,  
    true_mu=truemu_scaled_trace, true_Sigma=trueSigma_scaled_trace, true_Psi=truepsi,
    true_pi=trueclusterprob)                                        

_S = len(_sel)

T_grid_pos = np.arange(50, _S + 1)   
eval_pos   = np.arange(0, _S) 

blockcov_trace = block_coverage_trace(_cond, _chains, _truth, T_grid_pos, eval_pos,
                                      cat_dims=categorical_columns_dims)
T_grid_iters = _sel[T_grid_pos - 1]        

print('block coverage trace computed at', len(T_grid_pos), 'final-iteration values')
print('  mu coverage at the largest T:',    blockcov_trace['mu'][:, -1])
print('  Sigma coverage at the largest T:', blockcov_trace['Sigma'][:, -1])
print('  Psi coverage at the largest T (K x p):', blockcov_trace['Psi'][:, :, -1]) 
print('  pi coverage at the largest T:',    blockcov_trace['pi'][-1])          

_bcf = open('blockcov'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump({'burn_in': B_cov,                  
             'T_grid_iters': T_grid_iters,      
             'mu': blockcov_trace['mu'],        
             'Sigma': blockcov_trace['Sigma'],  
             'Psi': blockcov_trace['Psi'],      
             'pi': blockcov_trace['pi'],        

             'mu_ld_true': blockcov_trace['mu_ld_true'], 'mu_thr': blockcov_trace['mu_thr'],           
             'Sigma_ld_true': blockcov_trace['Sigma_ld_true'], 'Sigma_thr': blockcov_trace['Sigma_thr'], 
             'Psi_ld_true': blockcov_trace['Psi_ld_true'], 'Psi_thr': blockcov_trace['Psi_thr'],       
             'pi_ld_true': blockcov_trace['pi_ld_true'], 'pi_thr': blockcov_trace['pi_thr'],           
             'mu_ld_eval_last': blockcov_trace['mu_ld_eval_last'],                                     
             'Sigma_ld_eval_last': blockcov_trace['Sigma_ld_eval_last'],                               
             'Psi_ld_eval_last': blockcov_trace['Psi_ld_eval_last'],                                   
             'pi_ld_eval_last': blockcov_trace['pi_ld_eval_last'],                                    

             'mu_ld_true_mcse': blockcov_trace['mu_ld_true_mcse'],                                     
             'Sigma_ld_true_mcse': blockcov_trace['Sigma_ld_true_mcse'],                               
             'Psi_ld_true_mcse': blockcov_trace['Psi_ld_true_mcse'],                                   
             'pi_ld_true_mcse': blockcov_trace['pi_ld_true_mcse']},                                    
            _bcf)
_bcf.close()

end_timecov = time.time()
elapsed_timecov = end_timecov - start_timecov
