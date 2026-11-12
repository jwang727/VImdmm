import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from collections import Counter

import math
import csv

from scipy.stats import multivariate_normal, invwishart, wishart, dirichlet, rv_discrete, beta, gamma

import sys
print(sys.version)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataseedtrain", type=int, required=True)
parser.add_argument("--N", type=int, required=True)
parser.add_argument("--q", type=int, required=True)
parser.add_argument("--Ktrue", type=int, required=True)
args = parser.parse_args()

dataseedtrain = args.dataseedtrain
N = args.N
q= args.q
Ktrue = args.Ktrue
print(f"Running with dataseedtrain = {dataseedtrain} and N = {N} and q ={q} and Ktrue={Ktrue}")

p=1 

d_1=q 

maxdist=Ktrue*4 
maxvar=(maxdist*0.4)**2 

param_rng = np.random.default_rng(seed=13579) 

def checkdist(new_point, points, r_threshold):
    for point in points:
        dist = np.sqrt(np.sum(np.square(new_point-point)))
        if dist < r_threshold:
            return False


    return True


def Randvec(Nvec, L, r_threshold):
    points = []
    scope = np.arange(0,L,0.1)
    while len(points) < Nvec:
        new_point = param_rng.choice(scope, q)
        print('old_points')
        print(points)
        print('new_point')
        print(new_point)
        if checkdist(new_point, points, r_threshold):
            points.append(new_point)

    return points


def Randmat(maxvar):

    sampleunif=param_rng.uniform(low = maxvar/2, high = maxvar, size = q)
    sampleunif=np.round(sampleunif, decimals=1)

    from scipy.stats import ortho_group  

    U = ortho_group.rvs(dim=q, random_state=param_rng)

    return U @ np.diag(sampleunif) @ np.transpose(U)


def multipleRandmat(Nmat,maxvar):
    matriceslist=[]

    while len(matriceslist) < Nmat:
        new_mat = Randmat(maxvar)
        matriceslist.append(new_mat)
    return matriceslist


truemu_kg=np.zeros([Ktrue,1,q])

for k in range(0,Ktrue):

        truemu_kg[k,0,k]=maxdist

        print('k is '+str(k))


trueSigma_k=np.zeros([Ktrue,1,q,q])

for k in range(0,Ktrue):

        trueSigma_k[k,0,:]=multipleRandmat(1,maxvar)[0]


trueclusterprob = param_rng.uniform(1 / (2 * Ktrue), 2 / Ktrue, Ktrue)
trueclusterprob /= trueclusterprob.sum()

categorysizes=[d_1] 
truepsi=np.zeros([Ktrue,p]+[d_1]) 

highprob=0.75
truepsi[:,0,:]=(1-highprob)/(d_1-1)+np.zeros((Ktrue,d_1))
for k in range(0,Ktrue):
    truepsi[k,0,k] = highprob


def generate_data_contcase(N,Ktrue,truepsi,truemu_kg,trueSigma_k,trueclusterprob, data_seed):

    data_rng = np.random.default_rng(seed=data_seed)

    component = data_rng.choice(Ktrue, size=N, p=trueclusterprob)

    categoricaldata = np.zeros((N,p), dtype=int)

    for i in range(N):
        for j in range(p):

            categoricaldata[i, j] = data_rng.choice(np.arange(categorysizes[j]), p=truepsi[component[i],j,:] )


    continuousdata=np.zeros([N,q])

    for i in range(0,N):

        x = data_rng.multivariate_normal(truemu_kg[component[i],0,:], trueSigma_k[component[i],0,:], size=1)

        continuousdata[i,:]=x


    return continuousdata, categoricaldata, component


continuousdata, categoricaldata, component=generate_data_contcase(N,Ktrue,truepsi,truemu_kg,trueSigma_k,trueclusterprob, data_seed=dataseedtrain)

fulldatamat=np.hstack((continuousdata,np.reshape(categoricaldata,(N,p)), np.reshape(component,(N,1)) ))

contcol_list = [f'contcol_{i}' for i in range(1, q+1)]
catcol_list=[f'catcol_{i}' for i in range(1,p+1)]

fulldata=pd.DataFrame(fulldatamat, columns = contcol_list+catcol_list+['clusterlabel'])   

discrete_cols=catcol_list+['clusterlabel']

for col in discrete_cols:
    fulldata[col] = fulldata[col].astype(int)


allcategoricalvalues=[0,1]

modelled_continuous_columns=contcol_list 
modelled_categorical_columns=catcol_list

def merge_columns(row):
    return '_'.join(row.astype(str))


fulldata['combinedcategories']=fulldata[modelled_categorical_columns].apply(merge_columns, axis=1)

modelled_categorical_columns_combined=['combinedcategories']

fulldataframe=pd.DataFrame(fulldata, columns = modelled_continuous_columns+modelled_categorical_columns+modelled_categorical_columns_combined+['clusterlabel'])

scaler = StandardScaler()

fulldataframe[modelled_continuous_columns] = scaler.fit_transform(fulldataframe[modelled_continuous_columns])

scaler.fit_transform(np.array(fulldata[modelled_continuous_columns])) 

combinedcategories=np.array(fulldataframe['combinedcategories'])

continuousdata=np.array(fulldataframe[modelled_continuous_columns])

allcatvals=np.unique( np.array([str(i) for i in np.arange(d_1)], dtype='object') , return_counts=True) 

fulldataframegroupbymeans = fulldataframe.groupby(['combinedcategories'], as_index=False)[modelled_continuous_columns].mean()

fulldataframegroupbystd = fulldataframe.groupby(['combinedcategories'], as_index=False)[modelled_continuous_columns].std()

categorical_columns_dims=[]

for catcols in modelled_categorical_columns:

    categorical_columns_dims.append(d_1)


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

testcontinuousdata, testcategoricaldata, testcomponent=generate_data_contcase(Ntest,Ktrue,truepsi,truemu_kg,trueSigma_k,trueclusterprob, data_seed=dataseedtrain+1000)

testdatamat=np.hstack((testcontinuousdata,np.reshape(testcategoricaldata,(Ntest,p)), np.reshape(testcomponent,(Ntest,1)) ))

contcol_list = [f'contcol_{i}' for i in range(1, q+1)]

testdata=pd.DataFrame(testdatamat, columns = contcol_list+catcol_list+['clusterlabel'])   

for col in catcol_list+['clusterlabel']:
    testdata[col] = testdata[col].astype(int)


testdata['combinedcategories'] = testdata[modelled_categorical_columns].apply(merge_columns, axis=1)

testdataframe=pd.DataFrame(testdata, columns = modelled_continuous_columns+modelled_categorical_columns+modelled_categorical_columns_combined+['clusterlabel'])

testdataframe[modelled_continuous_columns] = scaler.transform(testdataframe[modelled_continuous_columns].to_numpy()) 

testcontinuousdata=np.array(testdataframe[modelled_continuous_columns])

print('done')

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

outputscaler=open('scaler'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl','wb')
pickle.dump(scaler,outputscaler)
outputscaler.close()

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

K = Ktrue 

iterations = 3000 

mdims = tuple([iterations,K] + [len(allcatvalsmu[0])] + [q])
betadims=tuple([iterations,K] + [len(allcatvalsmu[0])] )

m_kg_hat=np.zeros(mdims)
beta_kg_hat=np.zeros(betadims)

Phidims=tuple([iterations, K, len(allcatvalsSigma[0]), q, q])

nudims=tuple([iterations, K, len(allcatvalsSigma[0])])

Phi_k_hat=np.zeros(Phidims) 
nu_k_hat=np.zeros(nudims)

piparamdims=tuple([iterations]+ [K, len(allcatvalspi[0])])

piparam=np.zeros(piparamdims)

if Psiindependentofk==1:
    Psiparamdims=tuple([iterations, 1, p] + [np.max(categorical_columns_dims)])
else:
    Psiparamdims=tuple([iterations, K, p] + [np.max(categorical_columns_dims)])

Psiparam=np.zeros(Psiparamdims)

zparamdims=tuple([N, K])

r_ik=np.zeros(zparamdims)+0.1

rho_ik=np.zeros(zparamdims)
logrho_ik=np.zeros(zparamdims)

from kmodes.kprototypes import KPrototypes

cat_indices = [fulldataframe[modelled_continuous_columns+modelled_categorical_columns].columns.get_loc(col) for col in modelled_categorical_columns] 

kproto = KPrototypes(n_clusters=K, init='Cao', n_jobs=-1, random_state=42)
kprotoclusters = kproto.fit_predict(fulldataframe[modelled_continuous_columns+modelled_categorical_columns].to_numpy(), categorical=cat_indices)

for i in range(0,N):
    r_ik[i,kprotoclusters[i]]=0.9


print('done')

mu_kg_prior=np.zeros([K] + [len(allcatvalsmu[0])] + [q])

print('check mu_kg_prior')

m_kg_perturbation=np.random.multivariate_normal(mean=np.zeros(q),cov=0.25*np.eye(q),size=K)

if mudependentong==0:
    for k in range(0,K):
        for g in allcatvalsmu[0]:

            mu_kg_prior[k,catvaluetoindexmu[g],:] = np.zeros(q)

            print('k='+str(k)+' g='+g)
            print(mu_kg_prior[k,catvaluetoindexmu[g],:])


else:
    for k in range(0,K):
        for g in allcatvalsmu[0]:
            mu_kg_prior[k,catvaluetoindexmu[g],:] = np.zeros(q)

            print('k='+str(k)+' g='+g)
            print(mu_kg_prior[k,catvaluetoindexmu[g],:])


beta_kg_prior=1.0 

Phi=0.25*np.eye(q) 
nu=q + K + 1

piparamprior=1/K 

Psiparamprior=np.zeros(tuple([K, p] + [np.max(categorical_columns_dims)]))

for k in range(0,K):
    for j in range(0,p):
        for g_j in range(0,categorical_columns_dims[j]):
            Psiparamprior[k,j,g_j]=1/categorical_columns_dims[j]


print('check Psiparamprior')
print(Psiparamprior)
for k in range(0,K):
    for j in range(0,p):
        print('k='+str(k)+' j='+str(j))


print('done')

from scipy.special import digamma
from scipy.special import multigammaln, gammaln
from scipy.special import logsumexp
from scipy.special import xlogy
from scipy.special import softmax
import pickle
import time

ELBOoldcontonly=0

ELBO=1+np.zeros(iterations)
ELBOold=-10000
improvement=100
iteration=0

Elnpi_k=np.zeros((K,len(allcatvalspi[0])))
Elnpsi_kjg=np.zeros(tuple([K, p] + [np.max(categorical_columns_dims)]))
ElndetLambda_k=np.zeros((K,len(allcatvalsSigma[0])))
ElndetLambda_k_digammaterm=np.zeros((K,len(allcatvalsSigma[0])))

def quadform(vector,matrix):
    return np.transpose(vector) @ matrix @ vector


start_time = time.time()
vi_compute_elapsed = 0.0
timerecord = np.zeros(1000)

locations_pi_g = {g: np.where(combinedcategoriespi == g)[0] for g in allcatvalspi[0]}
locations_mu_g = {g: np.where(combinedcategoriesmu == g)[0] for g in allcatvalsmu[0]}
locations_Sigma_g = {g: np.where(combinedcategoriesSigma == g)[0] for g in allcatvalsSigma[0]}
locations_cat_jg = {}
counts_cat_jg = {}
for j in range(0,p):
    for g_j in range(0,categorical_columns_dims[j]):
        locations_cat_jg[(j,g_j)] = np.where(fulldataframe[modelled_categorical_columns[j]] == g_j)[0]
        counts_cat_jg[(j,g_j)] = np.sum(fulldataframe[modelled_categorical_columns[j]] == g_j)


sigma_indices = np.array([catvaluetoindexSigma[c] for c in combinedcategoriesSigma])
mu_indices = np.array([catvaluetoindexmu[c] for c in combinedcategoriesmu])
pi_indices = np.array([catvaluetoindexpi[c] for c in combinedcategoriespi])

unique_sigma_indices, sigma_inverse_map = np.unique(sigma_indices, return_inverse=True)
unique_mu_indices, mu_inverse_map = np.unique(mu_indices, return_inverse=True)

categoricaldataframe = {
col: fulldataframe[col].to_numpy()
for col in modelled_categorical_columns
}

log_inv_det_Phi_prior = np.log(1/np.linalg.det(Phi))

while iteration<1000 and (improvement>0.0001 or improvement<0): 

    iter_t0 = time.perf_counter()

    sum_r_pi_gk = {g: np.array([np.sum(r_ik[locations_pi_g[g],k]) for k in range(0,K)]) for g in allcatvalspi[0]}
    totsum_r_pi_g = {g: np.sum(r_ik[locations_pi_g[g],:], axis=None) for g in allcatvalspi[0]}
    sum_r_mu_gk = {g: np.array([np.sum(r_ik[locations_mu_g[g],k]) for k in range(0,K)]) for g in allcatvalsmu[0]}
    sum_r_Sigma_gk = {g: np.array([np.sum(r_ik[locations_Sigma_g[g],k]) for k in range(0,K)]) for g in allcatvalsSigma[0]}
    sum_r_cat_jgk = {key: np.array([np.sum(r_ik[idx,k]) for k in range(0,K)]) for key, idx in locations_cat_jg.items()}
    colsum_r_k = np.array([np.sum(r_ik[:,k]) for k in range(0,K)])

    for k in range(0,K):
        for g in allcatvalspi[0]:

            piparam[iteration,k,catvaluetoindexpi[g]]=piparamprior+sum_r_pi_gk[g][k]


    for k in range(0,K):
        for g in allcatvalsmu[0]:
            locations_g_idx = locations_mu_g[g] 

            beta_kg_hat[iteration,k,catvaluetoindexmu[g]]=beta_kg_prior+sum_r_mu_gk[g][k] 

            m_kg_hat[iteration,k,catvaluetoindexmu[g],:]=(beta_kg_prior*mu_kg_prior[k,catvaluetoindexmu[g],:]+np.transpose(continuousdata[locations_g_idx,:]) @ r_ik[locations_g_idx,k])/(beta_kg_hat[iteration,k,catvaluetoindexmu[g]])


    Phi_update_mhatttermstore=np.zeros((K,q,q))
    Phi_update_mtermstore=np.zeros((K,q,q))
    Phi_update_xtermstore=np.zeros((K,q,q))

    for k in range(0,K):

        for g in allcatvalsSigma[0]:
            locations_g_idx = locations_Sigma_g[g] 

            nu_k_hat[iteration,k,catvaluetoindexSigma[g]]=nu+sum_r_Sigma_gk[g][k] 

            m_kg_hat_vec=m_kg_hat[iteration,k,catvaluetoindexmu[g],:] 
            m_kg_hat_vec=np.reshape(m_kg_hat_vec,(m_kg_hat_vec.size,1))

            mu_kg_prior_vec=mu_kg_prior[k,catvaluetoindexmu[g],:]
            mu_kg_prior_vec=np.reshape(mu_kg_prior_vec,(mu_kg_prior_vec.size,1))

            Phi_update_mhattterm=(beta_kg_prior+sum_r_Sigma_gk[g][k])*m_kg_hat_vec @ np.transpose(m_kg_hat_vec) 

            Phi_update_mterm=beta_kg_prior*mu_kg_prior_vec @ np.transpose(mu_kg_prior_vec)

            X_g = continuousdata[locations_g_idx,:]
            r_g = r_ik[locations_g_idx,k]
            Phi_update_xterm = np.transpose(X_g) @ (r_g[:,np.newaxis]*X_g)

            Phi_k_hat[iteration,k,catvaluetoindexSigma[g],:]=Phi-Phi_update_mhattterm+Phi_update_mterm+Phi_update_xterm


        Phi_update_mhatttermstore[k,:]=Phi_update_mhattterm
        Phi_update_mtermstore[k,:]=Phi_update_mterm
        Phi_update_xtermstore[k,:]=Phi_update_xterm


    if Psiindependentofk==1:
        for j in range(0,p):
            for g_j in range(0,categorical_columns_dims[j]):

                Psiparam[iteration,0,j,g_j]=Psiparamprior[0,j,g_j] + counts_cat_jg[(j,g_j)] 


    else: 

        for k in range(0,K):
            for j in range(0,p):
                for g_j in range(0,categorical_columns_dims[j]):

                    Psiparam[iteration,k,j,g_j]=Psiparamprior[k,j,g_j]+sum_r_cat_jgk[(j,g_j)][k]


    ElndetLambda_k_digammaterm=np.zeros((K,len(allcatvalsSigma[0])))
    for k in range(0,K):
        for g in allcatvalsSigma[0]:
            for i in range(1,q+1): 
                ElndetLambda_k_digammaterm[k,catvaluetoindexSigma[g]]=ElndetLambda_k_digammaterm[k,catvaluetoindexSigma[g]]+digamma((nu_k_hat[iteration,k,catvaluetoindexSigma[g]]+1-i)/2) 


    for k in range(0,K):
        for g in allcatvalspi[0]:

            Elnpi_k[k,catvaluetoindexpi[g]]=digamma(piparamprior+sum_r_pi_gk[g][k])-digamma(K*piparamprior+totsum_r_pi_g[g])


    log_inv_det_Phi_k_hat=np.zeros((K,len(allcatvalsSigma[0])))
    inv_Phi_k_hat_stored=np.zeros((K,len(allcatvalsSigma[0]),q,q))
    for k in range(0,K):
        for g in allcatvalsSigma[0]:
            log_inv_det_Phi_k_hat[k,catvaluetoindexSigma[g]]=np.log(1/np.linalg.det(Phi_k_hat[iteration,k,catvaluetoindexSigma[g],:]))
            inv_Phi_k_hat_stored[k,catvaluetoindexSigma[g],:]=np.linalg.inv(Phi_k_hat[iteration,k,catvaluetoindexSigma[g],:])


    for k in range(0,K):
        for g in allcatvalsSigma[0]:

            ElndetLambda_k[k,catvaluetoindexSigma[g]]=q*np.log(2)+log_inv_det_Phi_k_hat[k,catvaluetoindexSigma[g]]+ElndetLambda_k_digammaterm[k,catvaluetoindexSigma[g]] 


    if Psiindependentofk==1:
        for j in range(0,p):
            for g_j in range(0,categorical_columns_dims[j]):

                Elnpsi_kjg[0,j,g_j]=digamma(Psiparamprior[0,j,g_j]+counts_cat_jg[(j,g_j)])-digamma(np.sum(Psiparamprior[0,j,:])+N) 


    else:

        for k in range(0,K):
            for j in range(0,p):
                for g_j in range(0,categorical_columns_dims[j]):

                    Elnpsi_kjg[k,j,g_j]=digamma(Psiparamprior[k,j,g_j]+sum_r_cat_jgk[(j,g_j)][k])-digamma(np.sum(Psiparamprior[k,j,:])+colsum_r_k[k]) 


    ElnNx_all_ik = np.zeros((N, K))

    ELBOElnNx_i = 0
    for k in range(0, K):

        unique_inv_Phi = np.linalg.inv(Phi_k_hat[iteration, k, unique_sigma_indices, :, :])

        unique_nu = nu_k_hat[iteration, k, unique_sigma_indices] 
        unique_beta = beta_kg_hat[iteration, k, unique_mu_indices]
        unique_m = m_kg_hat[iteration, k, unique_mu_indices, :]
        unique_ElndetLambda = ElndetLambda_k[k, unique_sigma_indices]

        nu_values = unique_nu[sigma_inverse_map]
        beta_values = unique_beta[mu_inverse_map]
        m_vectors = unique_m[mu_inverse_map] 
        ElndetLambda_values = unique_ElndetLambda[sigma_inverse_map]

        term1 = -(q / 2) * np.log(2*np.pi)
        term2 = 0.5 * ElndetLambda_values

        diffs = continuousdata - m_vectors

        quad_base = np.empty(diffs.shape[0])                                   
        for Phi_index in range(unique_inv_Phi.shape[0]):                               
            sub_Phi = (sigma_inverse_map == Phi_index)                                    
            diffs_sub = diffs[sub_Phi]                                                
            quad_base[sub_Phi] = ((diffs_sub @ unique_inv_Phi[Phi_index]) * diffs_sub).sum(axis=1)  


        quad_form_values = nu_values * quad_base                              

        term3 = -0.5 * quad_form_values
        term4 = -0.5 * q / beta_values

        ElnNx_all_i = term1 + term2 + term3 + term4
        ELBOElnNx_i += np.sum(r_ik[:, k] * ElnNx_all_i)

        ElnNx_all_ik[:, k] = ElnNx_all_i 


    ELBOElnpc = 0

    for j in range(0, p):                                                             
        _G = Elnpsi_kjg[:, j, categoricaldataframe[modelled_categorical_columns[j]]]   
        ELBOElnpc += np.sum(r_ik * _G.T)                                              


    ELBOElnpz = 0
    for k in range(0, K):
        term_for_k = np.sum(r_ik[:, k] * Elnpi_k[k, pi_indices])
        ELBOElnpz += term_for_k 


    ELBOElnqz = np.sum(xlogy(r_ik, r_ik))

    KLmugivenLambda=0
    for k in range(0,K):
        for g in allcatvalsmu[0]:

            KLmugivenLambdaincrement=-q*np.log(beta_kg_prior)+q*np.log(beta_kg_hat[iteration,k,catvaluetoindexmu[g]])-q+quadform(m_kg_hat[iteration,k,catvaluetoindexmu[g],:]-mu_kg_prior[k,catvaluetoindexmu[g],:],beta_kg_prior*nu_k_hat[iteration,k,catvaluetoindexmu[g]]*inv_Phi_k_hat_stored[k,catvaluetoindexSigma[g],:])+q*beta_kg_prior/beta_kg_hat[iteration,k,catvaluetoindexSigma[g]]
            KLmugivenLambda=KLmugivenLambda+0.5*KLmugivenLambdaincrement


    KLLambdadigterm=ElndetLambda_k_digammaterm

    KLLambda=0
    for k in range(0,K):
        for g in allcatvalsSigma[0]:

            KLLambdaincrement=nu*log_inv_det_Phi_prior-nu*log_inv_det_Phi_k_hat[k,catvaluetoindexSigma[g]]+nu_k_hat[iteration,k,catvaluetoindexSigma[g]]*np.trace(Phi @ inv_Phi_k_hat_stored[k,catvaluetoindexSigma[g],:] )+2*multigammaln(nu/2, q)-2*multigammaln(nu_k_hat[iteration,k,catvaluetoindexSigma[g]]/2, q)+(nu_k_hat[iteration,k,catvaluetoindexSigma[g]]-nu)*KLLambdadigterm[k,catvaluetoindexSigma[g]]-nu_k_hat[iteration,k,catvaluetoindexSigma[g]]*q
            KLLambda=KLLambda+0.5*KLLambdaincrement


    KLpi=0
    KLpiKsum=np.zeros(len(allcatvalspi[0])) 

    for k in range(0,K):
        for g in allcatvalspi[0]:
            KLpiKsum[catvaluetoindexpi[g]]=KLpiKsum[catvaluetoindexpi[g]]+gammaln(piparamprior)-gammaln(piparamprior+sum_r_pi_gk[g][k]) 


    KLpiNKsum=np.zeros(len(allcatvalspi[0])) 
    for k in range(0,K):
        for g in allcatvalspi[0]:
            KLpiNKsum[catvaluetoindexpi[g]]=KLpiNKsum[catvaluetoindexpi[g]]+sum_r_pi_gk[g][k]*Elnpi_k[k,catvaluetoindexpi[g]]


    for g in allcatvalspi[0]:
        KLpi=KLpi+gammaln(K*piparamprior+totsum_r_pi_g[g])-gammaln(K*piparamprior)+KLpiKsum[catvaluetoindexpi[g]]+KLpiNKsum[catvaluetoindexpi[g]] 


    if Psiindependentofk==1:
        KLPsi=0
        for j in range(0,p):

                KLPsiterm1=gammaln(np.sum(Psiparam[iteration,0,j,0:categorical_columns_dims[j]]))-gammaln(np.sum(Psiparamprior[0,j,0:categorical_columns_dims[j]]))
                KLPsiterm2=np.sum(gammaln(Psiparam[iteration,0,j,0:categorical_columns_dims[j]]))-np.sum(gammaln(Psiparamprior[0,j,0:categorical_columns_dims[j]]))
                KLPsiterm3=(Psiparam[iteration,0,j,0:categorical_columns_dims[j]]-Psiparamprior[0,j,0:categorical_columns_dims[j]])*( digamma(Psiparam[iteration,0,j,0:categorical_columns_dims[j]])-digamma(np.sum(Psiparam[iteration,0,j,0:categorical_columns_dims[j]])) )
                KLPsi=KLPsi+KLPsiterm1-KLPsiterm2+np.sum(KLPsiterm3) 


    else:
        KLPsi=0
        for k in range(0,K):
            for j in range(0,p):

                    KLPsiterm1=gammaln(np.sum(Psiparam[iteration,k,j,0:categorical_columns_dims[j]]))-gammaln(np.sum(Psiparamprior[k,j,0:categorical_columns_dims[j]]))
                    KLPsiterm2=np.sum(gammaln(Psiparam[iteration,k,j,0:categorical_columns_dims[j]]))-np.sum(gammaln(Psiparamprior[k,j,0:categorical_columns_dims[j]]))
                    KLPsiterm3=(Psiparam[iteration,k,j,0:categorical_columns_dims[j]]-Psiparamprior[k,j,0:categorical_columns_dims[j]])*( digamma(Psiparam[iteration,k,j,0:categorical_columns_dims[j]])-digamma(np.sum(Psiparam[iteration,k,j,0:categorical_columns_dims[j]])) )
                    KLPsi=KLPsi+KLPsiterm1-KLPsiterm2+np.sum(KLPsiterm3) 


    print('ELBO terms check')

    ELBO[iteration]=ELBOElnNx_i+ELBOElnpc+ELBOElnpz-ELBOElnqz-KLmugivenLambda-KLLambda-KLpi-KLPsi
    print('ELBO')
    print(ELBOElnNx_i+ELBOElnpc+ELBOElnpz-ELBOElnqz-KLmugivenLambda-KLLambda-KLpi-KLPsi)

    improvement=ELBOElnNx_i+ELBOElnpc+ELBOElnpz-ELBOElnqz-KLmugivenLambda-KLLambda-KLpi-KLPsi-ELBOold

    print('improvement')
    print(improvement)

    ELBOold=ELBOElnNx_i+ELBOElnpc+ELBOElnpz-ELBOElnqz-KLmugivenLambda-KLLambda-KLpi-KLPsi

    Elnpsisum_all_ik = np.zeros((N, K))
    Elnpi_all = np.zeros((N, K))
    for k in range(0, K):

        Elnpi_all[:, k] = Elnpi_k[k, pi_indices]


    if Psiindependentofk != 1:                                                        
        for j in range(0, p):                                                         
            Elnpsisum_all_ik += Elnpsi_kjg[:, j,                                       
                                           categoricaldataframe[modelled_categorical_columns[j]]].T   


    if Psiindependentofk == 1:
        logrho_ik = ElnNx_all_ik + Elnpi_all
    else:
        logrho_ik = ElnNx_all_ik + Elnpsisum_all_ik + Elnpi_all

    r_ik = softmax(logrho_ik, axis=1)

    vi_compute_elapsed += time.perf_counter() - iter_t0
    timerecord[iteration] = vi_compute_elapsed 

    print('sum of the r_ik by k')
    print(np.sum(r_ik, axis=0))

    print('predicted hard clusters')
    print(Counter(np.argmax(r_ik, axis=1)))

    print('true clusters')
    print(Counter(fulldataframe['clusterlabel']))

    iteration=iteration+1

end_time = time.time()
elapsed_time = end_time - start_time

print('elapsed_time')
print(elapsed_time)

iteration=iteration-1

import pickle

outputpiparam = open('piparam'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(piparam, outputpiparam)
outputpiparam.close()

outputbeta_kg_hat = open('beta_kg_hat'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(beta_kg_hat, outputbeta_kg_hat)
outputbeta_kg_hat.close()

outputm_kg_hat = open('m_kg_hat'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')

pickle.dump(m_kg_hat[max(iteration-11,0):iteration+1,:], outputm_kg_hat)
outputm_kg_hat.close()

outputnu_k_hat = open('nu_k_hat'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(nu_k_hat, outputnu_k_hat)
outputnu_k_hat.close()

outputPhi_k_hat = open('Phi_k_hat'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')

pickle.dump(Phi_k_hat[max(iteration-11,0):iteration+1,:], outputPhi_k_hat)

outputPhi_k_hat.close()

outputPsiparam = open('Psiparam'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(Psiparam, outputPsiparam)
outputPsiparam.close()

outputr_ik = open('r_ik'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(r_ik, outputr_ik)
outputr_ik.close()

outputELBO = open('ELBO'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(ELBO, outputELBO)
outputELBO.close()

outputelapsed_time = open('elapsed_time'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(elapsed_time, outputelapsed_time)
outputelapsed_time.close()

outputiteration = open('iteration'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(iteration, outputiteration)
outputiteration.close()

outputtimerecord = open('timerecord'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(timerecord, outputtimerecord)
outputtimerecord.close()

print('done')

from scipy.stats import multivariate_normal, invwishart, wishart, dirichlet, rv_discrete, beta, gamma, multivariate_t

def evalpostpred(x,c):
    g=str(c)
    mcsum=0
    for sample in range(0,nosamples):
        for k in range(0,K):
            piterm=pisamples[sample,k,catvaluetoindexpi[g]]
            normalterm=multivariate_normal.pdf(x,mean=musamples[sample,k,catvaluetoindexmu[g],:],cov=Sigmasamples[sample,k,catvaluetoindexSigma[g],:])
            psiterm=1
            for j in range(0,p):

                psiterm=psiterm*Psisamples[sample,k,j,int(g.split('_')[j])] 


            mcsum=mcsum+piterm*normalterm*psiterm


    return mcsum/nosamples


def evalpostpredanalytic(x,c):
    g=str(c)
    ksum=0
    for k in range(0,K):
        piterm=piparam[iteration,k,catvaluetoindexpi[g]]/np.sum(piparam[iteration,:,catvaluetoindexpi[g]])
        shapemattop=Phi_k_hat[iteration,k,catvaluetoindexSigma[g],:]*(beta_kg_hat[iteration,k,catvaluetoindexmu[g]]+1)
        shapematbottom=beta_kg_hat[iteration,k,catvaluetoindexmu[g]]*(nu_k_hat[iteration,k,catvaluetoindexSigma[g]]-q+1)
        normalterm=multivariate_t.pdf(x,loc=m_kg_hat[iteration,k,catvaluetoindexmu[g],:],shape=shapemattop/shapematbottom,df=nu_k_hat[iteration,k,catvaluetoindexSigma[g]]-q+1 )

        psiterm=1
        for j in range(0,p):

            psiterm=psiterm*Psiparam[iteration,k,j,int(g.split('_')[j])]/np.sum(Psiparam[iteration,k,j,0:categorical_columns_dims[j]]) 


        ksum=ksum+piterm*normalterm*psiterm


    return ksum


def evalpostpredtrue(x,c):
    g=str(c)
    ksum=0
    for k in range(0,Ktrue):
        piterm=trueclusterprob[k]

        normalterm=multivariate_normal.pdf(x,mean=scaler.transform(truemu_kg[k,catvaluetoindexmu[g]].reshape(1, -1))[0],cov=np.diag(1/scaler.scale_) @ trueSigma_k[k,catvaluetoindexmu[g],:] @ np.diag(1/scaler.scale_))

        psiterm=1
        for j in range(0,p):
            psiterm=psiterm*truepsi[k,j,int(g.split('_')[j])]


        ksum=ksum+piterm*normalterm*psiterm


    return ksum


posteriorpred=np.zeros(Ntest)
posteriorpredtrue=np.zeros(Ntest)
posteriorpredanalytic=np.zeros(Ntest)

for i in range(0,Ntest):

    posteriorpredtrue[i]=-np.log(evalpostpredtrue(testcontinuousdata[i],testdataframe['combinedcategories'].iloc[i]) )
    posteriorpredanalytic[i]=-np.log(evalpostpredanalytic(testcontinuousdata[i],testdataframe['combinedcategories'].iloc[i]) )

    print('i is '+str(i))

    print(posteriorpredtrue[i])
    print(posteriorpredanalytic[i])


print('maximum error')
posteriorpredmaxerror=np.max(np.abs(posteriorpredanalytic-posteriorpredtrue))
print(posteriorpredmaxerror)
print('mean error')
posteriorpredmeanerror=np.mean(np.abs(posteriorpredanalytic-posteriorpredtrue))
print(posteriorpredmeanerror)
print('mean order of magnitude for reference')
posteriorpredmean=np.mean(np.abs(posteriorpredtrue))

print(posteriorpredmean)

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


mu_error_vec=np.zeros(len(allcatvalsmu[0]))
mu_row_ind_vec=np.zeros((len(allcatvalsmu[0]),Ktrue))
mu_col_ind_vec=np.zeros((len(allcatvalsmu[0]),Ktrue))

sigma_error_vec=np.zeros(len(allcatvalsmu[0]))
sigma_row_ind_vec=np.zeros((len(allcatvalsmu[0]),Ktrue))
sigma_col_ind_vec=np.zeros((len(allcatvalsmu[0]),Ktrue))

print('mean in the original scale')
for g in allcatvalsmu[0]:
    print('g is '+g)
    error_permuted(truemu_kg[:,catvaluetoindexmu[g],:], scaler.inverse_transform(m_kg_hat[iteration,:,catvaluetoindexmu[g],:]) )


print('mean in the standardised scale')
for g in allcatvalsmu[0]:
    print('g is '+g)
    mu_error, mu_row_ind, mu_col_ind, matched_true_mu, matched_estimated_mu=error_permuted(scaler.transform(truemu_kg[:,catvaluetoindexmu[g],:]), m_kg_hat[iteration,:,catvaluetoindexmu[g],:])

    mu_error_vec[catvaluetoindexmu[g]]=mu_error
    mu_row_ind_vec[catvaluetoindexmu[g],:]=mu_row_ind 
    mu_col_ind_vec[catvaluetoindexmu[g],:]=mu_col_ind


print('covariance in the original scale')
for g in allcatvalsmu[0]:
    print('g is '+g)
    nu_k_hat_denom=nu_k_hat[iteration,:,catvaluetoindexSigma[g]]-q-1

    estimatedcovogscale=np.array([ np.diag(scaler.scale_) @ Phi_k_hat[iteration,k,catvaluetoindexSigma[g],:]/nu_k_hat_denom[k, np.newaxis, np.newaxis] @ np.diag(scaler.scale_) for k in range(K)])

    error_nopermute(trueSigma_k[:,catvaluetoindexmu[g],:], estimatedcovogscale, mu_row_ind_vec[catvaluetoindexmu[g],:].astype(int).tolist(), mu_col_ind_vec[catvaluetoindexmu[g],:].astype(int).tolist() ) 


print('covariance in the standardised scale')
for g in allcatvalsmu[0]:
    print('g is '+g)
    nu_k_hat_denom=nu_k_hat[iteration,:,catvaluetoindexSigma[g]]-q-1

    trueSigma_kscaled=np.array([ np.diag(1/scaler.scale_) @ trueSigma_k[k, catvaluetoindexmu[g], :] @ np.diag(1/scaler.scale_) for k in range(Ktrue)])

    sigma_error, sigma_row_ind, sigma_col_ind, matched_true_sigma, matched_estimated_sigma=error_nopermute(trueSigma_kscaled, Phi_k_hat[iteration,:,catvaluetoindexSigma[g],:]/nu_k_hat_denom[:, np.newaxis, np.newaxis], mu_row_ind_vec[catvaluetoindexmu[g],:].astype(int).tolist(), mu_col_ind_vec[catvaluetoindexmu[g],:].astype(int).tolist() )

    sigma_error_vec[catvaluetoindexmu[g]]=sigma_error
    sigma_row_ind_vec[catvaluetoindexmu[g],:]=sigma_row_ind
    sigma_col_ind_vec[catvaluetoindexmu[g],:]=sigma_col_ind


data_absmean=np.mean(np.abs(np.array(fulldataframe[modelled_continuous_columns])))

print(mu_error_vec)
print(mu_row_ind_vec)
print(mu_col_ind_vec)

print(type(mu_col_ind))

print(sigma_error_vec)
print(sigma_row_ind_vec)
print(sigma_col_ind_vec)

print(np.mean(mu_error_vec))
print(np.mean(sigma_error_vec))

print(np.mean(np.abs(scaler.transform(truemu_kg[:,catvaluetoindexmu[g],:]))))

print(data_absmean)

mapping = dict(zip(mu_col_ind, mu_row_ind))

predictedclusters=np.argmax(r_ik, axis=1)

orderedclusters=np.vectorize(lambda x: mapping.get(x, -1))(predictedclusters)

print('proportion of correct hardclusters')
print(np.sum(orderedclusters == component)/len(orderedclusters) )

hardclustermatchesprob=np.sum(orderedclusters == component)/len(orderedclusters)

predpsiprob=np.zeros((Ktrue,p,d_1))

Psi_error=np.zeros(p)

for j in range(0,p):
    print('we are at column '+str(j))

    Psiparamsum=np.sum(Psiparam[iteration,mu_col_ind,j,:], axis=1)
    print('truepsi')
    print( truepsi[:,j,:] )
    print('rearranged Psiparam prob')
    print( Psiparam[iteration,mu_col_ind,j,:]/Psiparamsum[:,np.newaxis] )

    print('error')
    Psi_error[j]=np.mean( np.abs(truepsi[:,j,:]-Psiparam[iteration,mu_col_ind,j,:]/Psiparamsum[:,np.newaxis]) )

    print(Psi_error[j])


print('Psi error averaged over p columns')
Psi_error_mean=np.mean(Psi_error)
print(Psi_error_mean)

print(np.shape(piparam))

print(trueclusterprob)

print(piparam[iteration,mu_col_ind,0]/np.sum(piparam[iteration,mu_col_ind,0])) 

print('Pi error')
estimateclusterprob=piparam[iteration,mu_col_ind,0]/np.sum(piparam[iteration,mu_col_ind,0])

print(np.abs(trueclusterprob-estimateclusterprob))

pi_error=np.mean(np.abs(trueclusterprob-estimateclusterprob)) 

print(pi_error)

import pickle

outputposteriorpredmean = open('posteriorpredmean'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(posteriorpredmean, outputposteriorpredmean)
outputposteriorpredmean.close()

outputposteriorpredmeanerror = open('posteriorpredmeanerror'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(posteriorpredmeanerror, outputposteriorpredmeanerror)
outputposteriorpredmeanerror.close()

outputdata_absmean=open('data_absmean'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(data_absmean, outputdata_absmean)
outputdata_absmean.close()

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

print('done')

nosamples=2000

sample_rng = np.random.default_rng(seed=42) 

pisamples=np.zeros([nosamples,K,len(allcatvalspi[0])])

print(np.shape(pisamples))

logposterior=np.zeros(nosamples)

logposteriorpi=np.zeros(nosamples)
logposteriorSigma=np.zeros(nosamples)
logposteriormu=np.zeros(nosamples)
logposteriorPsi=np.zeros(nosamples)

logposteriorSigma_perk=np.zeros((nosamples,K))
logposteriormu_perk=np.zeros((nosamples,K))
logposteriorPsi_perk=np.zeros((nosamples,K))

for sample in range(0,nosamples):
    for g in allcatvalspi[0]:
        pisamples[sample,:,catvaluetoindexpi[g]]=dirichlet.rvs(alpha=piparam[iteration,:,catvaluetoindexpi[g]], size=1, random_state=sample_rng)
        logposterior[sample]=logposterior[sample]+dirichlet.logpdf(pisamples[sample,:,catvaluetoindexpi[g]],alpha=piparam[iteration,:,catvaluetoindexpi[g]])

        logposteriorpi[sample]=logposteriorpi[sample]+dirichlet.logpdf(pisamples[sample,:,catvaluetoindexpi[g]],alpha=piparam[iteration,:,catvaluetoindexpi[g]])


print('done pisamples')
print('check mean and 5% percentile of logposterior after pi')
print(np.mean(logposterior))
print(np.percentile(logposterior, 5))

Sigmasamples=np.zeros([nosamples,K,len(allcatvalsSigma[0]),q,q])

print(np.shape(Sigmasamples))

for sample in range(0,nosamples):
    for k in range(0,K):
        for g in allcatvalsSigma[0]:
            Sigmasamples[sample,k,catvaluetoindexSigma[g],:]=invwishart.rvs(df=nu_k_hat[iteration,k,catvaluetoindexSigma[g]], scale=Phi_k_hat[iteration,k,catvaluetoindexSigma[g],:], size=1, random_state=sample_rng)
            logposterior[sample]=logposterior[sample]+invwishart.logpdf(Sigmasamples[sample,k,catvaluetoindexSigma[g],:], df=nu_k_hat[iteration,k,catvaluetoindexSigma[g]], scale=Phi_k_hat[iteration,k,catvaluetoindexSigma[g],:])

            logposteriorSigma[sample]=logposteriorSigma[sample]+invwishart.logpdf(Sigmasamples[sample,k,catvaluetoindexSigma[g],:], df=nu_k_hat[iteration,k,catvaluetoindexSigma[g]], scale=Phi_k_hat[iteration,k,catvaluetoindexSigma[g],:])

            logposteriorSigma_perk[sample,k]=logposteriorSigma_perk[sample,k]+invwishart.logpdf(Sigmasamples[sample,k,catvaluetoindexSigma[g],:], df=nu_k_hat[iteration,k,catvaluetoindexSigma[g]], scale=Phi_k_hat[iteration,k,catvaluetoindexSigma[g],:])


print('done Sigmasamples')
print('check mean and 5% percentile of logposterior after Sigma')
print(np.mean(logposterior))
print(np.percentile(logposterior, 5))

musamples=np.zeros([nosamples,K,len(allcatvalsmu[0]),q])

print(np.shape(musamples))

for sample in range(0,nosamples):
    for k in range(0,K):
        for g in allcatvalsmu[0]:
            musamples[sample,k,catvaluetoindexmu[g],:]=multivariate_normal.rvs(mean=m_kg_hat[iteration,k,catvaluetoindexmu[g],:], cov=Sigmasamples[sample,k,catvaluetoindexSigma[g],:]/beta_kg_hat[iteration,k,catvaluetoindexmu[g]], size=1, random_state=sample_rng)
            logposterior[sample]=logposterior[sample]+multivariate_normal.logpdf(musamples[sample,k,catvaluetoindexmu[g],:], mean=m_kg_hat[iteration,k,catvaluetoindexmu[g],:], cov=Sigmasamples[sample,k,catvaluetoindexSigma[g],:]/beta_kg_hat[iteration,k,catvaluetoindexmu[g]])

            tdistSigma=Phi_k_hat[iteration,k,catvaluetoindexSigma[g],:]/(nu_k_hat[iteration,k,catvaluetoindexSigma[g]]-q+1)
            tdistSigma=tdistSigma/beta_kg_hat[iteration,k,catvaluetoindexmu[g]]
            logposteriormu[sample]=logposteriormu[sample]+multivariate_t.logpdf(musamples[sample,k,catvaluetoindexmu[g],:], loc=m_kg_hat[iteration,k,catvaluetoindexmu[g],:], shape=tdistSigma, df=nu_k_hat[iteration,k,catvaluetoindexSigma[g]]-q+1)

            logposteriormu_perk[sample,k]=logposteriormu_perk[sample,k]+multivariate_t.logpdf(musamples[sample,k,catvaluetoindexmu[g],:], loc=m_kg_hat[iteration,k,catvaluetoindexmu[g],:], shape=tdistSigma, df=nu_k_hat[iteration,k,catvaluetoindexSigma[g]]-q+1)


print('done musamples')
print('check mean and 5% percentile of logposterior after mu')
print(np.mean(logposterior))
print(np.percentile(logposterior, 5))

Psisamples=np.zeros([nosamples,K,p,np.max(categorical_columns_dims)])
print(np.shape(Psisamples))

for sample in range(0,nosamples):
    for k in range(0,K):
        for j in range(0,p):
            Psisamples[sample,k,j,0:categorical_columns_dims[j]]=dirichlet.rvs(alpha=Psiparam[iteration,k,j,0:categorical_columns_dims[j]], random_state=sample_rng)
            logposterior[sample]=logposterior[sample]+dirichlet.logpdf(Psisamples[sample,k,j,0:categorical_columns_dims[j]],
                                                                       alpha=Psiparam[iteration,k,j,0:categorical_columns_dims[j]])

            logposteriorPsi[sample]=logposteriorPsi[sample]+dirichlet.logpdf(Psisamples[sample,k,j,0:categorical_columns_dims[j]],
                                                                             alpha=Psiparam[iteration,k,j,0:categorical_columns_dims[j]])

            logposteriorPsi_perk[sample,k]=logposteriorPsi_perk[sample,k]+dirichlet.logpdf(Psisamples[sample,k,j,0:categorical_columns_dims[j]],
                                                                                           alpha=Psiparam[iteration,k,j,0:categorical_columns_dims[j]])


print('done Psisamples')
print('check mean and 5% percentile of logposterior after Psi')
print(np.mean(logposterior))
print(np.percentile(logposterior, 5))

d=d_1

logposteriortrue=0.0

logposteriortruepi=0.0
logposteriortrueSigma=0.0
logposteriortruemu=0.0
logposteriortruePsi=0.0

logposteriortrueSigma_perk=np.zeros(K)
logposteriortruemu_perk=np.zeros(K)
logposteriortruePsi_perk=np.zeros(K)

trueclusterprobfull=np.zeros(K)
trueclusterprobfull[mu_col_ind]=trueclusterprob

truemu_kgfull=np.zeros((K,q))
truemu_kgfull[mu_col_ind,:]=scaler.transform(truemu_kg[:,0,:])

trueSigma_full=np.zeros((K,q,q))
trueSigma_full[mu_col_ind,:]=np.array([ np.diag(1/scaler.scale_) @ trueSigma_k[k, 0, :] @ np.diag(1/scaler.scale_) for k in range(Ktrue)]) 

truepsi_full=np.zeros((K,p,d))
for j in range(0,p):
    truepsi_full[mu_col_ind,j,:]=truepsi[:,j,:]


for g in allcatvalspi[0]:
    print('pi posterior mean')
    print(piparam[iteration,:,0]/np.sum(piparam[iteration,:,0]) )

    print('true pi')
    print(trueclusterprobfull)

    logposteriortrue=logposteriortrue+dirichlet.logpdf(trueclusterprobfull,alpha=piparam[iteration,:,catvaluetoindexpi[g]])

    logposteriortruepi=logposteriortruepi+dirichlet.logpdf(trueclusterprobfull,alpha=piparam[iteration,:,catvaluetoindexpi[g]])


print('check logposteriortrue after pi')
print(logposteriortrue)        

for k in range(0,K):
    for g in allcatvalsSigma[0]:
        print('k is '+str(k))
        nu_k_hat_denom=nu_k_hat[iteration,:,catvaluetoindexSigma[g]]-q-1
        print('Sigma posterior mean')
        print(Phi_k_hat[iteration,k,catvaluetoindexSigma[g],:]/nu_k_hat_denom[k] )
        print('Sigma true value')
        print(trueSigma_full[k,:])

        logposteriortrue=logposteriortrue+invwishart.logpdf(trueSigma_full[k,:], df=nu_k_hat[iteration,k,catvaluetoindexSigma[g]], scale=Phi_k_hat[iteration,k,catvaluetoindexSigma[g],:])

        logposteriortrueSigma=logposteriortrueSigma+invwishart.logpdf(trueSigma_full[k,:], df=nu_k_hat[iteration,k,catvaluetoindexSigma[g]], scale=Phi_k_hat[iteration,k,catvaluetoindexSigma[g],:])

        logposteriortrueSigma_perk[k]=logposteriortrueSigma_perk[k]+invwishart.logpdf(trueSigma_full[k,:], df=nu_k_hat[iteration,k,catvaluetoindexSigma[g]], scale=Phi_k_hat[iteration,k,catvaluetoindexSigma[g],:])


print('check logposteriortrue after Sigma')
print(logposteriortrue)       

for k in range(0,K):
    for g in allcatvalsmu[0]:
        print('k is '+str(k))
        print('mu posterior mean')
        print(m_kg_hat[iteration,k,catvaluetoindexmu[g],:] )
        print('mu true value')
        print(truemu_kgfull[k,:])

        logposteriortrue=logposteriortrue+multivariate_normal.logpdf(truemu_kgfull[k,:], mean=m_kg_hat[iteration,k,catvaluetoindexmu[g],:], cov=trueSigma_full[k,:]/beta_kg_hat[iteration,k,catvaluetoindexmu[g]])

        tdistSigma=Phi_k_hat[iteration,k,catvaluetoindexSigma[g],:]/(nu_k_hat[iteration,k,catvaluetoindexSigma[g]]-q+1)
        tdistSigma=tdistSigma/beta_kg_hat[iteration,k,catvaluetoindexmu[g]]
        logposteriortruemu=logposteriortruemu+multivariate_t.logpdf(truemu_kgfull[k,:], loc=m_kg_hat[iteration,k,catvaluetoindexmu[g],:], shape=tdistSigma, df=nu_k_hat[iteration,k,catvaluetoindexSigma[g]]-q+1)

        logposteriortruemu_perk[k]=logposteriortruemu_perk[k]+multivariate_t.logpdf(truemu_kgfull[k,:], loc=m_kg_hat[iteration,k,catvaluetoindexmu[g],:], shape=tdistSigma, df=nu_k_hat[iteration,k,catvaluetoindexSigma[g]]-q+1)


print('check logposteriortrue after mu')
print(logposteriortrue)             

for k in range(0,K):
    for j in range(0,p):
        print('k is '+str(k)+' j is '+str(j))
        print('psi posterior mean')
        Psiparamsum=np.sum(Psiparam[iteration,:,j,:], axis=1)
        print(Psiparam[iteration,k,j,:]/Psiparamsum[k])
        print('psi true value')
        print(truepsi_full[k,j,:])

        logposteriortrue=logposteriortrue+dirichlet.logpdf(truepsi_full[k,j,:],alpha=Psiparam[iteration,k,j,0:categorical_columns_dims[j]])

        logposteriortruePsi=logposteriortruePsi+dirichlet.logpdf(truepsi_full[k,j,:],alpha=Psiparam[iteration,k,j,0:categorical_columns_dims[j]])

        logposteriortruePsi_perk[k]=logposteriortruePsi_perk[k]+dirichlet.logpdf(truepsi_full[k,j,:],
                                                                                 alpha=Psiparam[iteration,k,j,0:categorical_columns_dims[j]])


print('check logposteriortrue after psi')
print(logposteriortrue)             

print('done calculating logposteriortrue')

print('total and individual HDI 5 percentile')
print(np.percentile(logposterior, 5))
print(np.percentile(logposteriorpi, 5))
print(np.percentile(logposteriorSigma, 5))
print(np.percentile(logposteriormu, 5))
print(np.percentile(logposteriorPsi, 5))

print('total and individual true 5 percentile')
print(logposteriortrue)
print(logposteriortruepi)
print(logposteriortrueSigma)
print(logposteriortruemu)
print(logposteriortruePsi)

logposteriorall= [np.percentile(logposterior, 5), np.percentile(logposteriorpi, 5), np.percentile(logposteriorSigma, 5), np.percentile(logposteriormu, 5), np.percentile(logposteriorPsi, 5)]
logposteriortrueall = [logposteriortrue, logposteriortruepi, logposteriortrueSigma, logposteriortruemu, logposteriortruePsi]
coverageresults = [a_true > a for a_true, a in zip(logposteriortrueall, logposteriorall)]

print(logposteriorall)
print(logposteriortrueall)
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

print(posteriorpredmean)

print(posteriorpredmeanerror)

print(data_absmean)

print(mu_error_vec)

print(sigma_error_vec)

print(hardclustermatchesprob)

print(Psi_error_mean)

print(pi_error)

logposteriorSigma_perk_5pct=np.array([np.percentile(logposteriorSigma_perk[:,k], 5) for k in range(0,K)])
logposteriormu_perk_5pct=np.array([np.percentile(logposteriormu_perk[:,k], 5) for k in range(0,K)])
logposteriorPsi_perk_5pct=np.array([np.percentile(logposteriorPsi_perk[:,k], 5) for k in range(0,K)])

coverageresultsSigma_perk=[logposteriortrueSigma_perk[k] > logposteriorSigma_perk_5pct[k] for k in range(0,K)]
coverageresultsmu_perk=[logposteriortruemu_perk[k] > logposteriormu_perk_5pct[k] for k in range(0,K)]
coverageresultsPsi_perk=[logposteriortruePsi_perk[k] > logposteriorPsi_perk_5pct[k] for k in range(0,K)]

print('per-k Sigma HDI 5 percentile / true / coverage')
print(logposteriorSigma_perk_5pct)
print(logposteriortrueSigma_perk)
print(coverageresultsSigma_perk)
print('per-k mu HDI 5 percentile / true / coverage')
print(logposteriormu_perk_5pct)
print(logposteriortruemu_perk)
print(coverageresultsmu_perk)
print('per-k Psi HDI 5 percentile / true / coverage')
print(logposteriorPsi_perk_5pct)
print(logposteriortruePsi_perk)
print(coverageresultsPsi_perk)

logposteriorall_perk=[logposteriorSigma_perk_5pct, logposteriormu_perk_5pct, logposteriorPsi_perk_5pct]
logposteriortrueall_perk=[logposteriortrueSigma_perk, logposteriortruemu_perk, logposteriortruePsi_perk]
coverageresults_perk=[coverageresultsSigma_perk, coverageresultsmu_perk, coverageresultsPsi_perk]

outputlogposteriorall_perk=open('logposteriorall_perk'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(logposteriorall_perk, outputlogposteriorall_perk)
outputlogposteriorall_perk.close()

outputlogposteriortrueall_perk=open('logposteriortrueall_perk'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(logposteriortrueall_perk, outputlogposteriortrueall_perk)
outputlogposteriortrueall_perk.close()

outputcoverageresults_perk=open('coverageresults_perk'+str(dataseedtrain)+'N'+str(N)+'q'+str(q)+'Ktrue'+str(Ktrue)+'.pkl', 'wb')
pickle.dump(coverageresults_perk, outputcoverageresults_perk)
outputcoverageresults_perk.close()
