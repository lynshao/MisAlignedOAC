import numpy as np
import pdb


def per_pkt_transmission(args, MM, TransmittedSymbols):
    # Pass the channel and generate samples at the receiver
    taus = np.sort(np.random.uniform(0,args.maxDelay,(1,MM-1)))[0]
    taus[-1] = args.maxDelay
    dd = np.zeros(MM)
    for idx in np.arange(MM):
        if idx == 0:
            dd[idx] = taus[0]
        elif idx == MM-1:
            dd[idx] = 1 - taus[-1]
        else:
            dd[idx] = taus[idx] - taus[idx-1]

    # # # Generate the channel: phaseOffset = 0->0; 1->2pi/4; 2->2pi/2; 3->2pi
    if args.phaseOffset == 0:
        hh = np.ones([MM,1])
    elif args.phaseOffset == 1:
        hh = np.exp(1j*np.random.uniform(0,1,(MM,1)) * 2* np.pi/4)
    elif args.phaseOffset == 2:
        hh = np.exp(1j*np.random.uniform(0,1,(MM,1)) * 3* np.pi/4)
    else:
        hh = np.exp(1j*np.random.uniform(0,1,(MM,1)) * 4* np.pi /4)

    # complex pass the complex channel
    for idx in range(MM):
        TransmittedSymbols[idx,:] = TransmittedSymbols[idx,:] * hh[idx][0]

    # compute the received signal power and add noise
    LL = len(TransmittedSymbols[0])
    SignalPart = np.sum(TransmittedSymbols,0)
    SigPower = np.sum(np.power(np.abs(SignalPart),2))/LL
    # SigPower = np.max(np.power(np.abs(SignalPart),2))
    EsN0 = np.power(10, args.EsN0dB/10.0)
    noisePower = SigPower/EsN0

    # Oversample the received signal
    RepeatedSymbols = np.repeat(TransmittedSymbols, MM, axis = 1)
    for idx in np.arange(MM):
        extended = np.array([np.r_[np.zeros(idx), RepeatedSymbols[idx], np.zeros(MM-idx-1)]])
        if idx == 0:
            samples = extended
        else:
            samples = np.r_[samples, extended]
    samples = np.sum(samples, axis=0)
    
    # generate noise
    for idx in np.arange(MM):
        noise = np.random.normal(loc=0, scale=np.sqrt(noisePower/2/dd[idx]), size=LL+1)+1j*np.random.normal(loc=0, scale=np.sqrt(noisePower/2/dd[idx]), size=LL+1)
        if idx == 0:
            AWGNnoise = np.array([noise])
        else:
            AWGNnoise = np.r_[AWGNnoise, np.array([noise])]

    AWGNnoise = np.reshape(AWGNnoise, (1,MM*(LL+1)), 'F')
    samples = samples + AWGNnoise[0][0:-1]

    # aligned_sample estiamtor
    if args.Estimator == 1:
        MthFiltersIndex = (np.arange(LL) + 1) * MM - 1
        output = samples[MthFiltersIndex]
        return output/MM

    # ML estiamtor
    if args.Estimator == 2:
        noisePowerVec = noisePower/2./dd
        HH = np.zeros([MM*(LL+1)-1, MM*LL])
        for idx in range(MM*LL):
            HH[np.arange(MM)+idx, idx] = hh[np.mod(idx,MM)]
        CzVec = np.tile(noisePowerVec, [1, LL+1])
        Cz = np.diag(CzVec[0][:-1])
        ## ------------------------------------- ML
        MUD = np.matmul(HH.conj().T, np.linalg.inv(Cz))
        MUD = np.matmul(MUD, HH)
        MUD = np.matmul(np.linalg.inv(MUD), HH.conj().T)
        MUD = np.matmul(MUD, np.linalg.inv(Cz))
        MUD = np.matmul(MUD, np.array([samples]).T)

        ## ------------------------------------- Estimate SUM
        output = np.sum(np.reshape(MUD, [LL, MM]), 1)
        return output/MM

    # SP_ML estiamtor
    if args.Estimator == 3:
        noisePowerVec = noisePower/2./dd
        output = BP_Decoding(samples, MM, LL, hh, noisePowerVec)
        return output/MM

def BP_Decoding(samples, M, L, hh, noisePowerVec):
    # Prepare the Gaussian messages (Eta,LambdaMat) obtained from the observation nodes
    # Lambda
    beta1 = np.c_[np.real(hh),np.imag(hh)]
    beta2 = np.c_[-np.imag(hh),np.real(hh)]
    Obser_Lamb_first = np.c_[np.matmul(beta1,np.transpose(beta1)),np.matmul(beta1,np.transpose(beta2))]
    Obser_Lamb_second = np.c_[np.matmul(beta2,np.transpose(beta1)),np.matmul(beta1,np.transpose(beta1))]
    Obser_Lamb = np.r_[Obser_Lamb_first,Obser_Lamb_second]
    element = np.zeros([4,4])
    element[0,0] = 1
    ObserMat1 = np.tile(element,(2,2)) * Obser_Lamb # pos-by-pos multiplication
    element = np.zeros([4,4])
    element[0:2,0:2] = 1
    ObserMat2 = np.tile(element,(2,2)) * Obser_Lamb # pos-by-pos multiplication
    element = np.zeros([4,4])
    element[0:3,0:3] = 1
    ObserMat3 = np.tile(element,(2,2)) * Obser_Lamb # pos-by-pos multiplication
    element = np.ones([4,4])
    ObserMat4 = np.tile(element,(2,2)) * Obser_Lamb # pos-by-pos multiplication
    element = np.zeros([4,4])
    element[1:,1:] = 1
    ObserMat5 = np.tile(element,(2,2)) * Obser_Lamb # pos-by-pos multiplication
    element = np.zeros([4,4])
    element[2:,2:] = 1
    ObserMat6 = np.tile(element,(2,2)) * Obser_Lamb # pos-by-pos multiplication
    element = np.zeros([4,4])
    element[3:,3:] = 1
    ObserMat7 = np.tile(element,(2,2)) * Obser_Lamb # pos-by-pos multiplication

    # Eta = LambdaMat * mean
    etaMat = np.matmul(np.r_[beta1,beta2],np.r_[np.real([samples]),np.imag([samples])])

    # process the boundaries
    etaMat[[1,2,3,5,6,7],0] = 0
    etaMat[[2,3,6,7],1] = 0
    etaMat[[3,7],2] = 0
    etaMat[[0,4],-3] = 0
    etaMat[[0,1,4,5],-2] = 0
    etaMat[[0,1,2,4,5,6],-1] = 0

    # ============================================================
    # ============================================================
    # ============================================================ right message passing
    R_m3_eta = np.zeros([2*M, M*(L+1)-2])
    R_m3_Lamb = np.zeros([2*M, 2*M, M*(L+1)-2])
    for idx in range(M*(L+1)-2):
        # ----------------------------- message m1(eta,Lamb) from bottom
        m1_eta = etaMat[:,idx] / noisePowerVec[np.mod(idx,M)]
        if idx == 0: # first boundary -- will only be used in the right passing
            ObserMat = ObserMat1
        elif idx == 1: # second boundary
            ObserMat = ObserMat2
        elif idx == 2:# third boundary
            ObserMat = ObserMat3
        elif idx == M*(L+1)-4: # second last boundary
            ObserMat = ObserMat5
        elif idx == M*(L+1)-3: # second last boundary
            ObserMat = ObserMat6
        elif idx == M*(L+1)-2: # last boundary -- will only be used in the left passing
            ObserMat = ObserMat7
        else:
            ObserMat = ObserMat4
        m1_Lamb = ObserMat / noisePowerVec[np.mod(idx,M)]

        # ----------------------------- message m2: right message => product of bottom and left
        if idx == 0: # first boundary
            m2_eta = m1_eta
            m2_Lamb = m1_Lamb
        else:
            m2_eta = m1_eta + R_m3_eta[:,idx-1]
            m2_Lamb = m1_Lamb + R_m3_Lamb[:,:,idx-1]

        # ----------------------------- message m3: sum
        m2_Sigma = np.linalg.pinv(m2_Lamb) # find the matrix Sigma of m2
        pos = [np.mod(idx+1,M), np.mod(idx+1,M)+M] # pos of two variables (real and imag) to be integrated
        # convert m2_eta back to m2_mean to delete columns -> convert back and add zero columns -> get the new m3_eta
        m2_mean = np.matmul(m2_Sigma, m2_eta) # m2_mean
        m2_mean[pos] = 0 # set to zero and convert back to eta (see below)
        m2_Sigma[pos,:] = 0 # delete the rows and columns of m2_Sigma
        m2_Sigma[:,pos] = 0
        m3_Lamb = np.linalg.pinv(m2_Sigma)
        m3_eta = np.matmul(m3_Lamb, m2_mean)
        # ----------------------------- store m3
        R_m3_eta[:,idx] = m3_eta
        R_m3_Lamb[:,:,idx] = m3_Lamb

    # ============================================================
    # ============================================================
    # ============================================================ left message passing
    L_m3_eta = np.zeros([2*M, M*(L+1)-1])
    L_m3_Lamb = np.zeros([2*M, 2*M, M*(L+1)-1])

    for idx in np.arange(M*(L+1)-2, 0, -1):
        # ----------------------------- message m1: from bottom
        m1_eta = etaMat[:,idx] / noisePowerVec[np.mod(idx,M)];
        if idx == 0: # first boundary -- will only be used in the right passing
            ObserMat = ObserMat1
        elif idx == 1: # second boundary
            ObserMat = ObserMat2
        elif idx == 2: # third boundary
            ObserMat = ObserMat3
        elif idx == M*(L+1)-4: # second last boundary
            ObserMat = ObserMat5
        elif idx == M*(L+1)-3: # second last boundary
            ObserMat = ObserMat6
        elif idx == M*(L+1)-2: # last boundary -- will only be used in the left passing
            ObserMat = ObserMat7
        else:
            ObserMat = ObserMat4

        m1_Lamb = ObserMat / noisePowerVec[np.mod(idx,M)]

        # ----------------------------- message m2: product
        if idx == M*(L+1)-2: # last boundary
            m2_eta = m1_eta
            m2_Lamb = m1_Lamb
        else:
            m2_eta = m1_eta + L_m3_eta[:,idx+1]
            m2_Lamb = m1_Lamb + L_m3_Lamb[:,:,idx+1]

        # ----------------------------- message m3: sum
        m2_Sigma = np.linalg.pinv(m2_Lamb) # find the matrix Sigma of m2
        pos = [np.mod(idx,M), np.mod(idx,M)+M] # pos of two variables (real and imag) to be integrated
        # convert m2_eta back to m2_mean to delete columns -> convert back and add zero columns -> get the new m3_eta
        m2_mean = np.matmul(m2_Sigma, m2_eta) # m2_mean
        m2_mean[pos] = 0 # set to zero and convert back to eta (see below)
        # convert m2_Lambda back to m2_Sigma to delete rows/columns -> convert back and add zero rows/columns -> get the new m3_Lambda
        m2_Sigma[pos,:] = 0
        m2_Sigma[:,pos] = 0
        m3_Lamb = np.linalg.pinv(m2_Sigma)
        m3_eta = np.matmul(m3_Lamb, m2_mean)
        # ----------------------------- store m3
        L_m3_eta[:,idx] = m3_eta
        L_m3_Lamb[:,:,idx] = m3_Lamb

    # ------------------------- Marginalization & BP DECODING
    Sum_mu = np.zeros(L) + 1j * 0
    for ii in range(1, L+1):
        idx = ii * M - 1
        
        Res_Eta = etaMat[:, idx] / noisePowerVec[np.mod(idx,M)] + R_m3_eta[:,idx-1] + L_m3_eta[:,idx+1]
        Res_Lamb = ObserMat4 / noisePowerVec[np.mod(idx,M)] + R_m3_Lamb[:,:,idx-1] + L_m3_Lamb[:,:,idx+1]
        # Res_Eta = etaMat[:, idx] / noisePowerVec[np.mod(idx,M)]
        # Res_Lamb = ObserMat4 / noisePowerVec[np.mod(idx,M)]

        # compute (mu,Sigma) for a variable node
        Res_Sigma = np.linalg.pinv(Res_Lamb)
        Res_mu = np.matmul(Res_Sigma, Res_Eta)

        # compute (mu,Sigma) for the sum
        Sum_mu[ii-1] = np.sum(Res_mu[0:M]) + 1j *np.sum(Res_mu[M:])
    
    return Sum_mu

def test():
    from options import args_parser
    MM = 4
    LL = 1000
    args = args_parser()
    args.EsN0dB = 10

    # Generate TransmittedSymbols
    for m in range(MM):
        symbols = 2 * np.random.randint(2, size=(2,LL)) - 1
        ComplexSymbols = symbols[0,:] + symbols[1,:] * 1j
        if m == 0:
            TransmittedSymbols = np.array([ComplexSymbols])
        else:
            TransmittedSymbols = np.r_[TransmittedSymbols, np.array([ComplexSymbols])]
    
    target = np.sum(TransmittedSymbols, 0)
    # MSE of the aligned_sample estimator
    args.Estimator = 1
    output = per_pkt_transmission(args, MM, TransmittedSymbols)
    MSE1 = np.mean(np.power(np.abs(output - target),2))
    print('MSE1 = ', MSE1)

    # MSE of the ML estimator
    args.Estimator = 2
    output = per_pkt_transmission(args, MM, TransmittedSymbols)
    MSE2 = np.mean(np.power(np.abs(output - target),2))
    print('MSE2 = ', MSE2)

    # MSE of the SP-ML estimator
    args.Estimator = 4
    output = per_pkt_transmission(args, MM, TransmittedSymbols)
    MSE4 = np.mean(np.power(np.abs(output - target),2))
    print('MSE4 = ', MSE4)

if __name__ == "__main__":
    test()
