# Based on original code from Nicole Dumont

import numpy as np
import scipy.stats

def fractional_bind(basis, position):
    return np.fft.ifft(np.prod(np.fft.fft(basis, axis=0)**position, axis=1), axis=0).real
    
    
# Helper funstions 
def _get_sub_FourierSSP(n, N, sublen=3):
    # Return a matrix, \bar{A}_n
    # Consider the multi scale representation (S_{total}) and sub vectors (S_n) described in the paper 
    # Then
    # \bar{A}_n F{S_{total}} = F{S_n}
    # i.e. pick out the sub vector in the Fourier domain
    tot_len = 2*sublen*N + 1
    FA = np.zeros((2*sublen + 1, tot_len))
    FA[0:sublen, sublen*n:sublen*(n+1)] = np.eye(sublen)
    FA[sublen, sublen*N] = 1
    FA[sublen+1:, tot_len - np.arange(sublen*(n+1),sublen*n,-1)] = np.eye(sublen)
    return FA

def _get_sub_SSP(n,N,sublen=3):
    # Return a matrix, A_n
    # Consider the multi scale representation (S_{total}) and sub vectors (S_n) described in the paper 
    # Then
    # A_n S_{total} = S_n
    # i.e. pick out the sub vector in the time domain
    tot_len = 2*sublen*N + 1
    FA = _get_sub_FourierSSP(n,N,sublen=sublen)
    W = np.fft.fft(np.eye(tot_len))
    invW = np.fft.ifft(np.eye(2*sublen + 1))
    A = invW @ np.fft.ifftshift(FA) @ W
    return A.real

def _proj_sub_FourierSSP(n,N,sublen=3):
    # Return a matrix, \bar{B}_n
    # Consider the multi scale representation (S_{total}) and sub vectors (S_n) described in the paper 
    # Then
    # \sum_n \bar{B}_n F{S_{n}} = F{S_{total}}
    # i.e. project the sub vector in the Fourier domain such that summing all such projections gives the full vector in Fourier domain
    tot_len = 2*sublen*N + 1
    FB = np.zeros((2*sublen + 1, tot_len))
    FB[0:sublen, sublen*n:sublen*(n+1)] = np.eye(sublen)
    FB[sublen, sublen*N] = 1/N # all sub vectors have a "1" zero freq term so scale it so full vector will have 1 
    FB[sublen+1:, tot_len - np.arange(sublen*(n+1),sublen*n,-1)] = np.eye(sublen)
    return FB.T

def _proj_sub_SSP(n,N,sublen=3):
    # Return a matrix, B_n
    # Consider the multi scale representation (S_{total}) and sub vectors (S_n) described in the paper 
    # Then
    # \sum_n B_n S_{n} = S_{total}
    # i.e. project the sub vector in the time domain such that summing all such projections gives the full vector
    tot_len = 2*sublen*N + 1
    FB = _proj_sub_FourierSSP(n,N,sublen=sublen)
    invW = np.fft.ifft(np.eye(tot_len))
    W = np.fft.fft(np.eye(2*sublen + 1))
    B = invW @ np.fft.ifftshift(FB) @ W
    return B.real



class GridBasis:
    def __init__(self, dimensions, n_rotates=8, scales=np.linspace(0.5, 2, 8), hex=True):
        self.dimensions = dimensions
        self.n_rotates = n_rotates
        self.scales = scales
        self.hex = hex
        
        self.compute_phases()
        self.compute_basis()
        
        
    def compute_phases(self):
        dim = self.dimensions
        
        if self.hex:
            K = np.hstack([np.sqrt(1+ 1/dim)*np.identity(dim) - (dim**(-3/2))*(np.sqrt(dim+1) + 1),
                         (dim**(-1/2))*np.ones((dim,1))]).T
        else:
            K = np.identity(dim)
        K_scales = np.vstack([K*s for s in self.scales])
        if dim == 2 and self.n_rotates > 1:
            angles = np.linspace(0,2*np.pi/3, self.n_rotates)
            R_mats = np.stack([np.stack([np.cos(angles), -np.sin(angles)],axis=1),
                        np.stack([np.sin(angles), np.cos(angles)], axis=1)], axis=1)
            K_scale_rotates = (R_mats @ K_scales.T).transpose(0,2,1).reshape(-1,dim)
        elif dim>1 and self.n_rotates>1:
            R_mats = scipy.stats.special_ortho_group.rvs(dim, size=self.n_rotates)
            K_scale_rotates = (R_mats @ K_scales.T).transpose(0,2,1).reshape(-1,dim)
        else: 
            assert self.n_rotates == 1
            K_scale_rotates = K_scales        
        self.phases = K_scale_rotates
        
    # Using a matrix of phases to generate axis vectors
    def compute_basis(self):
        K = self.phases # K is a matrix of phases
        d = K.shape[0]
        n = K.shape[1]
        axes = []
        axes_fft = []
        for i in range(n):
            F = np.ones((d*2 + 1,), dtype="complex")
            F[0:d] = np.exp(1.j*K[:,i])
            F[-d:] = np.flip(np.conj(F[0:d]))
            F = np.fft.ifftshift(F)
            axes_fft.append(F)
            axes.append(np.fft.ifft(F).real)
        self.axes = np.array(axes)
        self.axes_fft = np.array(axes_fft)
        
    def encode(self, x):
        x = np.asarray(x)
        shape = x.shape
        x = x.reshape(-1, self.dimensions)
        r = np.fft.ifft(np.prod(np.power(self.axes_fft.T[:,:,None], x.T), axis=1), axis=0).T.real
        return r.reshape(shape[:-1]+(-1,))

    def make_encoders(self, n_G, radius=10):
        sub_dim = self.dimensions
        if self.hex:
            sub_dim += 1
        d = self.axes.shape[1]
        N = (d-1)//(2*sub_dim) 
        assert N == self.n_rotates * len(self.scales)
        G_pos = np.random.rand(n_G,self.dimensions)*2*radius - radius
        G_sorts = np.random.randint(0, N, size = n_G)
        G_encoders = np.zeros((n_G,d))
        for i in range(n_G):
            sub_mat = _get_sub_SSP(G_sorts[i],N,sublen=sub_dim)
            proj_mat = _proj_sub_SSP(G_sorts[i],N,sublen=sub_dim)
            basis_i = sub_mat @ self.axes.T
            G_encoders[i,:] = N * proj_mat @ fractional_bind(basis_i, G_pos[i,:])
        return G_encoders