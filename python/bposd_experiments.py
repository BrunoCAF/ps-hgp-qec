import numpy as np
import scipy.sparse as sp
from ldpc import mod2
from ldpc.alist import save_alist
from ldpc.code_util import compute_code_distance
from bposd import stab
from tqdm import tqdm
import json
import time
import datetime
from bposd import bposd_decoder

def row_echelon(matrix, full=False):
    """
    Converts a binary matrix to row echelon form via Gaussian Elimination

    Parameters
    ----------
    matrix : numpy.ndarray or scipy.sparse
        A binary matrix in either numpy.ndarray format or scipy.sparse
    full: bool, optional
        If set to 'True', Gaussian elimination is only performed on the rows below
        the pivot. If set to 'False' Gaussian eliminatin is performed on rows above
        and below the pivot. 
    
    Returns
    -------
        row_ech_form: numpy.ndarray
            The row echelon form of input matrix
        rank: int
            The rank of the matrix
        transform_matrix: numpy.ndarray
            The transformation matrix such that (transform_matrix@matrix)=row_ech_form
        pivot_cols: list
            List of the indices of pivot num_cols found during Gaussian elimination

    """
    num_rows, num_cols = np.shape(matrix)

    # Take copy of matrix if numpy (why?) and initialise transform matrix to identity
    if isinstance(matrix, np.ndarray):
        the_matrix = np.copy(matrix)
        transform_matrix = np.identity(num_rows).astype(int)
    elif isinstance(matrix, sp.csr.csr_matrix):
        the_matrix = matrix
        transform_matrix = sp.eye(num_rows, dtype="int", format="csr")
    else:
        raise ValueError('Unrecognised matrix type')

    pivot_row = 0
    pivot_cols = []

    # Iterate over cols, for each col find a pivot (if it exists)
    for col in range(num_cols):

        # Select the pivot - if not in this row, swap rows to bring a 1 to this row, if possible
        if the_matrix[pivot_row, col] != 1:

            # Find a row with a 1 in this col
            swap_row_index = pivot_row + np.argmax(the_matrix[pivot_row:num_rows, col])

            # If an appropriate row is found, swap it with the pivot. Otherwise, all zeroes - will loop to next col
            if the_matrix[swap_row_index, col] == 1:

                # Swap rows
                the_matrix[[swap_row_index, pivot_row]] = the_matrix[[pivot_row, swap_row_index]]

                # Transformation matrix update to reflect this row swap
                transform_matrix[[swap_row_index, pivot_row]] = transform_matrix[[pivot_row, swap_row_index]]

        # If we have got a pivot, now let's ensure values below that pivot are zeros
        if the_matrix[pivot_row, col]:

            if not full:  
                elimination_range = [k for k in range(pivot_row + 1, num_rows)]
            else:
                elimination_range = [k for k in range(num_rows) if k != pivot_row]

            # Let's zero those values below the pivot by adding our current row to their row
            for j in elimination_range:

                if the_matrix[j, col] != 0 and pivot_row != j:    ### Do we need second condition?

                    the_matrix[j] = (the_matrix[j] + the_matrix[pivot_row]) % 2

                    # Update transformation matrix to reflect this op
                    transform_matrix[j] = (transform_matrix[j] + transform_matrix[pivot_row]) % 2

            pivot_row += 1
            pivot_cols.append(col)

        # Exit loop once there are no more rows to search
        if pivot_row >= num_rows:
            break

    # The rank is equal to the maximum pivot index
    matrix_rank = pivot_row
    row_esch_matrix = the_matrix

    return [row_esch_matrix, matrix_rank, transform_matrix, pivot_cols]


class css_code():

    def __init__(self,hx=np.array([[]]),hz=np.array([[]]),code_distance=np.nan, name="<Unnamed CSS code>"):

        self.hx=hx #hx pcm
        self.hz=hz #hz pcm

        self.lx=np.array([[]]) #x logicals
        self.lz=np.array([[]]) #z logicals

        self.N=np.nan #block length
        self.K=np.nan #code dimension
        self.D=code_distance #code distance
        self.L=np.nan #max column weight
        self.Q=np.nan #max row weight

        _,nx=self.hx.shape
        _,nz=self.hz.shape

        assert nx==nz, "Error: hx and hz matrices must have equal numbers of columns!"

        if nx!=0:
            self.compute_dimension()
            self.compute_ldpc_params()
            self.compute_logicals()
            if code_distance==0:
                dx=compute_code_distance(hx)
                dz=compute_code_distance(hz)
                self.D=np.min([dx,dz])
                
        self.name=name

    def compute_dimension(self):

        self.N=self.hx.shape[1]
        assert self.N == self.hz.shape[1], "Code block length (N) inconsistent!"

        self.K=self.N-mod2.rank(self.hx)-mod2.rank(self.hz)
        return self.K

    def compute_ldpc_params(self):

        #column weights
        hx_l=np.max(np.sum(self.hx,axis=0))
        hz_l=np.max(np.sum(self.hz,axis=0))
        self.L=np.max([hx_l,hz_l]).astype(int)

        #row weights
        hx_q=np.max(np.sum(self.hx,axis=1))
        hz_q=np.max(np.sum(self.hz,axis=1))
        self.Q=np.max([hx_q,hz_q]).astype(int)

    def save_sparse(self, code_name):

        self.code_name=code_name

        hx=self.hx
        hz=self.hz
        save_alist(f"{code_name}_hx.alist",hx)
        save_alist(f"{code_name}_hz.alist",hz)

        lx=self.lx
        lz=self.lz
        save_alist(f"{code_name}_lx.alist",lx)
        save_alist(f"{code_name}_lz.alist",lz)

    def to_stab_code(self):

        hx=np.vstack([np.zeros(self.hz.shape,dtype=int),self.hx])
        hz=np.vstack([self.hz,np.zeros(self.hx.shape,dtype=int)])
        return stab.stab_code(hx,hz)

    @property
    def h(self):
        hx=np.vstack([np.zeros(self.hz.shape,dtype=int),self.hx])
        hz=np.vstack([self.hz,np.zeros(self.hx.shape,dtype=int)])
        return np.hstack([hx,hz])

    @property
    def l(self):
        lx=np.vstack([np.zeros(self.lz.shape,dtype=int),self.lx])
        lz=np.vstack([self.lz,np.zeros(self.lx.shape,dtype=int)])
        return np.hstack([lx,lz])


    def compute_code_distance(self):
        temp=self.to_stab_code()
        self.D=temp.compute_code_distance()
        return self.D

    def compute_logicals(self):

        def compute_lz(hx,hz):
            #lz logical operators
            #lz\in ker{hx} AND \notin Im(Hz.T)

            ker_hx=mod2.nullspace(hx) #compute the kernel basis of hx
            im_hzT=mod2.row_basis(hz) #compute the image basis of hz.T

            #in the below we row reduce to find vectors in kx that are not in the image of hz.T.
            log_stack=sp.vstack([im_hzT,ker_hx]).todense()
            
            pivots=row_echelon(log_stack.T)[3]
            log_op_indices=[i for i in range(im_hzT.shape[0],log_stack.shape[0]) if i in pivots]
            log_ops=log_stack[log_op_indices]
            return log_ops

        if np.isnan(self.K): self.compute_dimension()
        self.lx=compute_lz(self.hz,self.hx)
        self.lz=compute_lz(self.hx,self.hz)

        return self.lx,self.lz

    def canonical_logicals(self):
        temp=mod2.inverse(self.lx@self.lz.T %2)
        self.lx=temp@self.lx %2


    @property
    def code_params(self):
        try: self.N
        except AttributeError: self.N=np.nan
        try: self.K
        except AttributeError: self.K=np.nan
        try: self.D
        except AttributeError: self.D=np.nan
        try: self.L
        except AttributeError: self.L=np.nan
        try: self.Q
        except AttributeError: self.Q=np.nan

        return f"({self.L},{self.Q})-[[{self.N},{self.K},{self.D}]]"

    def test(self, show_tests=True):
        valid_code=True

        if np.isnan(self.K): self.compute_dimension()
        self.compute_ldpc_params()

        code_label=f"{self.code_params}"

        if show_tests: print(f"{self.name}, {code_label}")

        try:
            assert self.N==self.hz.shape[1]==self.lz.shape[1]==self.lx.shape[1]
            assert self.K==self.lz.shape[0]==self.lx.shape[0]
            if show_tests: print(" -Block dimensions: Pass")
        except AssertionError:
            valid_code=False
            print(" -Block dimensions incorrect")

        try:
            assert not (self.hz@self.hx.T %2).any()
            if show_tests: print(" -PCMs commute hz@hx.T==0: Pass")
        except AssertionError:
            valid_code=False
            print(" -PCMs commute hz@hx.T==0: Fail")

        try:
            assert not (self.hx@self.hz.T %2).any()
            if show_tests: print(" -PCMs commute hx@hz.T==0: Pass")
        except AssertionError:
            valid_code=False
            print(" -PCMs commute hx@hz.T==0: Fail")

        try:
            assert not (self.hz@self.lx.T %2).any()
        except AssertionError:
            valid_code=False
            print(r" -lx \in ker{hz} AND lz \in ker{hx}: Fail")


        try:
            assert not (self.hx@self.lz.T %2).any()
            if show_tests: print(r" -lx \in ker{hz} AND lz \in ker{hx}: Pass")
        except AssertionError:
            valid_code=False
            print(r" -lx \in ker{hz} AND lz \in ker{hx}: Fail")


        try:
            assert mod2.rank(self.lx@self.lz.T %2)==self.K
            if show_tests: print(" -lx and lz anticommute: Pass")
        except AssertionError:
            valid_code=False
            print(" -lx and lz anticommute: Fail")

        if show_tests and valid_code: print(f" -{self.name} is a valid CSS code w/ params {code_label}")

        return valid_code
    


class css_decode_sim():

    '''
    A class for simulating BP+OSD decoding of CSS codes

    Note
    ....
    The input parameters can be entered directly or as a dictionary. 

    Parameters
    ----------

    hx: numpy.ndarray
        The hx matrix of the CSS code.
    hz: numpy.ndarray
        The hz matrix of the CSS code.
    error_rate: float
        The physical error rate on each qubit.
    xyz_error_bias: list of ints
        The relative bias for X, Y and Z errors.
    seed: int
        The random number generator seed.
    target_runs: int
        The number of runs you wish to simulate.
    bp_method: string
        The BP method. Choose either: 1) "minimum_sum"; 2) "product_sum".
    ms_scaling_factor: float
        The minimum sum scaling factor (if applicable)
    max_iter: int
        The maximum number of iterations for BP.
    osd_method: string
        The OSD method. Choose from: 1) "osd_cs"; 2) "osd_e"; 3) "osd0".
    channel_update: string
        The channel update method. Choose form: 1) None; 2) "x->z"; 3) "z->x".
    output_file: string
        The output file to write to.
    save_interval: int
        The time in interval (in seconds) between writing to the output file.
    check_code: bool
        Check whether the CSS code is valid.
    tqdm_disable: bool
        Enable/disable the tqdm progress bar. If you are running this script on a HPC
        cluster, it is recommend to disable tqdm.
    run_sim: bool
        If enabled (default), the simulation will start automatically.
    hadamard_rotate: bool
        Toggle Hadamard rotate. ON: 1; OFF; 0
    hadamard_rotate_sector1_length: int
        Specifies the number of qubits in sector 1 for the Hadamard rotation.
    error_bar_precision_cutoff: float
        The simulation will stop after this precision is reached.
    '''

    def __init__(self, hx=None, hz=None, **input_dict):

        # default input values
        default_input = {
            'error_rate': None,
            'xyz_error_bias': [1, 1, 1],
            'target_runs': 100,
            'seed': 0,
            'bp_method': "minimum_sum",
            'ms_scaling_factor': 0.625,
            'max_iter': 0,
            'osd_method': "osd_cs",
            'osd_order': 2,
            'save_interval': 2,
            'output_file': None,
            'check_code': 0,
            'tqdm_disable': 0,
            'run_sim': 1,
            'channel_update': "x->z",
            'hadamard_rotate': 0,
            'hadamard_rotate_sector1_length': 0,
            'error_bar_precision_cutoff': 1e-3
        }

        #apply defaults for keys not passed to the class
        for key in input_dict.keys():
            self.__dict__[key] = input_dict[key]
        for key in default_input.keys():
            if key not in input_dict:
                self.__dict__[key] = default_input[key]

        # output variables
        output_values = {
            "K": None,
            "N": None,
            "start_date": None,
            "runtime": 0.0,
            "runtime_readable": None,
            "run_count": 0,
            "bp_converge_count_x": 0,
            "bp_converge_count_z": 0,
            "bp_success_count": 0,
            "bp_logical_error_rate": 0,
            "bp_logical_error_rate_eb": 0,
            "osd0_success_count": 0,
            "osd0_logical_error_rate": 0.0,
            "osd0_logical_error_rate_eb": 0.0,
            "osdw_success_count": 0,
            "osdw_logical_error_rate": 0.0,
            "osdw_logical_error_rate_eb": 0.0,
            "osdw_word_error_rate": 0.0,
            "osdw_word_error_rate_eb": 0.0,
            "min_logical_weight": 1e9
        }

        for key in output_values.keys(): #copies initial values for output attributes
            if key not in self.__dict__:
                self.__dict__[key] = output_values[key]

        #the attributes we wish to save to file
        temp = [] 
        for key in self.__dict__.keys():
            if key not in ['channel_probs_x','channel_probs_z','channel_probs_y','hx','hz']:
                temp.append(key)
        self.output_keys = temp

        #random number generator setup
        if self.seed==0 or self.run_count!=0:
            self.seed=np.random.randint(low=1,high=2**32-1)
        np.random.seed(self.seed)
        print(f"RNG Seed: {self.seed}")
        
        # the hx and hx matrices
        self.hx = hx.astype(int)
        self.hz = hz.astype(int)
        self.N = self.hz.shape[1] #the block length
        if self.min_logical_weight >= 1e9: #the minimum observed weight of a logical operator
            self.min_logical_weight=self.N 
        self.error_x = np.zeros(self.N).astype(int) #x_component error vector
        self.error_z = np.zeros(self.N).astype(int) #z_component error vector

        # construct the CSS code from hx and hz
        self._construct_code()

        # setup the error channel
        self._error_channel_setup()

        # setup the BP+OSD decoders
        self._decoder_setup()

        if self.run_sim:
            self.run_decode_sim()

    def _single_run(self):

        '''
        The main simulation procedure
        '''

        # randomly generate the error
        self.error_x, self.error_z = self._generate_error()

        if self.channel_update is None:
            # decode z
            synd_z = self.hx@self.error_z % 2
            self.bpd_z.decode(synd_z)

            # decode x
            synd_x = self.hz@self.error_x % 2
            self.bpd_x.decode(synd_x)

        elif self.channel_update=="z->x":
            # decode z
            synd_z = self.hx@self.error_z % 2
            self.bpd_z.decode(synd_z)

            self.bpd_z_bp_decoding = self.bpd_z.bp_decoding
            self.bpd_z_osd0_decoding = self.bpd_z.osd0_decoding
            self.bpd_z_osdw_decoding = self.bpd_z.osdw_decoding
            
            #update the channel probability
            self._channel_update(self.channel_update)

            # decode x
            synd_x = self.hz@self.error_x % 2
            self.bpd_x.decode(synd_x)

        elif self.channel_update=="x->z":
            
            # decode x
            synd_x = self.hz@self.error_x % 2
            self.bpd_x.decode(synd_x)

            self.bpd_x_bp_decoding = self.bpd_x.bp_decoding
            self.bpd_x_osd0_decoding = self.bpd_x.osd0_decoding
            self.bpd_x_osdw_decoding = self.bpd_x.osdw_decoding
            
            #update the channel probability
            self._channel_update(self.channel_update)

            # decode z
            synd_z = self.hx@self.error_z % 2
            self.bpd_z.decode(synd_z)

        self.synd_x = synd_x
        self.synd_z = synd_z

        self.bpd_x_bp_decoding = self.bpd_x.bp_decoding
        self.bpd_z_bp_decoding = self.bpd_z.bp_decoding
        self.bpd_x_osd0_decoding = self.bpd_x.osd0_decoding
        self.bpd_z_osd0_decoding = self.bpd_z.osd0_decoding
        self.bpd_x_osdw_decoding = self.bpd_x.osdw_decoding
        self.bpd_z_osdw_decoding = self.bpd_z.osdw_decoding
    
        #compute the logical and word error rates
        if not (self.synd_x.any() or self.synd_z.any()):
            self.bpd_x_bp_decoding *= 0
            self.bpd_z_bp_decoding *= 0
            self.bpd_x_osd0_decoding *= 0
            self.bpd_z_osd0_decoding *= 0
            self.bpd_x_osdw_decoding *= 0
            self.bpd_z_osdw_decoding *= 0

        self.bpd_x_converge = self.bpd_x.converge
        self.bpd_z_converge = self.bpd_z.converge
        
        if not self.synd_x.any():
            self.bpd_x_converge = True
        if not self.synd_z.any():
            self.bpd_z_converge = True
        
        self._encoded_error_rates()

    def _channel_update(self,update_direction):

        '''
        Function updates the channel probability vector for the second decoding component
        based on the first. The channel probability updates can be derived from Bayes' rule.
        '''

        #x component first, then z component
        if update_direction=="x->z":
            decoder_probs=np.zeros(self.N)
            for i in range(self.N):
                if self.bpd_x_osdw_decoding[i]==1:
                    if (self.channel_probs_x[i]+self.channel_probs_y[i])==0:
                        decoder_probs[i]=0
                    else:
                        decoder_probs[i]=self.channel_probs_y[i]/(self.channel_probs_x[i]+self.channel_probs_y[i])
                elif self.bpd_x.osdw_decoding[i]==0:
                        decoder_probs[i]=self.channel_probs_z[i]/(1-self.channel_probs_x[i]-self.channel_probs_y[i])
        
            self.bpd_z.update_channel_probs(decoder_probs)

        #z component first, then x component
        elif update_direction=="z->x":
            decoder_probs=np.zeros(self.N)
            for i in range(self.N):
                if self.bpd_z_osdw_decoding[i]==1:
                    
                    if (self.channel_probs_z[i]+self.channel_probs_y[i])==0:
                        decoder_probs[i]=0
                    else:
                        decoder_probs[i]=self.channel_probs_y[i]/(self.channel_probs_z[i]+self.channel_probs_y[i])
                elif self.bpd_z.osdw_decoding[i]==0:
                        decoder_probs[i]=self.channel_probs_x[i]/(1-self.channel_probs_z[i]-self.channel_probs_y[i])

            
            self.bpd_x.update_channel_probs(decoder_probs)

    def _encoded_error_rates(self):

        '''
        Updates the logical and word error rates for OSDW, OSD0 and BP (before post-processing)
        '''

        #OSDW Logical error rate
        # calculate the residual error
        residual_x = (self.error_x+self.bpd_x_osdw_decoding) % 2
        residual_z = (self.error_z+self.bpd_z_osdw_decoding) % 2

        # check for logical X-error
        if (self.lz@residual_x % 2).any():
            logical_weight = np.sum(residual_x)
            if logical_weight < self.min_logical_weight:
                self.min_logical_weight = int(logical_weight)

        # check for logical Z-error
        elif (self.lx@residual_z % 2).any():
            logical_weight = np.sum(residual_z)
            if logical_weight < self.min_logical_weight:
                self.min_logical_weight = int(logical_weight)
        else:
            self.osdw_success_count += 1

        # compute logical error rate
        self.osdw_logical_error_rate = 1-self.osdw_success_count/self.run_count
        self.osdw_logical_error_rate_eb = np.sqrt(
            (1-self.osdw_logical_error_rate)*self.osdw_logical_error_rate/self.run_count)

        # compute word error rate
        self.osdw_word_error_rate = 1.0 - \
            (1-self.osdw_logical_error_rate)**(1/self.K)
        self.osdw_word_error_rate_eb = self.osdw_logical_error_rate_eb * \
            ((1-self.osdw_logical_error_rate_eb)**(1/self.K - 1))/self.K

        #OSD0 logical error rate
        # calculate the residual error
        residual_x = (self.error_x+self.bpd_x_osd0_decoding) % 2
        residual_z = (self.error_z+self.bpd_z_osd0_decoding) % 2

        # check for logical X-error
        if (self.lz@residual_x % 2).any():
            logical_weight = np.sum(residual_x)
            if logical_weight < self.min_logical_weight:
                self.min_logical_weight = int(logical_weight)

        # check for logical Z-error
        elif (self.lx@residual_z % 2).any():
            logical_weight = np.sum(residual_z)
            if logical_weight < self.min_logical_weight:
                self.min_logical_weight = int(logical_weight)
        else:
            self.osd0_success_count += 1

        # compute logical error rate
        self.osd0_logical_error_rate = 1-self.osd0_success_count/self.run_count
        self.osd0_logical_error_rate_eb = np.sqrt(
            (1-self.osd0_logical_error_rate)*self.osd0_logical_error_rate/self.run_count)

        # compute word error rate
        self.osd0_word_error_rate = 1.0 - \
            (1-self.osd0_logical_error_rate)**(1/self.K)
        self.osd0_word_error_rate_eb = self.osd0_logical_error_rate_eb * \
            ((1-self.osd0_logical_error_rate_eb)**(1/self.K - 1))/self.K

        #BP Logical error rate
        #check for convergence
        if self.bpd_z_converge:
            self.bp_converge_count_z+=1
        if self.bpd_x_converge:
            self.bp_converge_count_x+=1

        if self.bpd_z_converge and self.bpd_x_converge:
            # calculate the residual error
            residual_x = (self.error_x+self.bpd_x_bp_decoding) % 2
            residual_z = (self.error_z+self.bpd_z_bp_decoding) % 2

            # check for logical X/Z-error
            if not ((self.lz@residual_x % 2).any() or (self.lx@residual_z % 2).any()):
                self.bp_success_count += 1

        # compute logical error rate
        self.bp_logical_error_rate = 1-self.bp_success_count/self.run_count
        self.bp_logical_error_rate_eb = np.sqrt(
            (1-self.bp_logical_error_rate)*self.bp_logical_error_rate/self.run_count)

        # compute word error rate
        self.bp_word_error_rate = 1.0 - \
            (1-self.bp_logical_error_rate)**(1/self.K)
        self.bp_word_error_rate_eb = self.bp_logical_error_rate_eb * \
            ((1-self.bp_logical_error_rate_eb)**(1/self.K - 1))/self.K

    def _construct_code(self):

        '''
        Constructs the CSS code from the hx and hz stabilizer matrices.
        '''

        # print("Constructing CSS code from hx and hz matrices...")
        assert isinstance(self.hx, np.ndarray) and isinstance(self.hz, np.ndarray)
        
        qcode = css_code(self.hx, self.hz)
        self.lx = qcode.lx
        self.lz = qcode.lz
        self.K = qcode.K
        self.N = qcode.N
        # print("Checking the CSS code is valid...")
        
        return None

    def _error_channel_setup(self):

        '''
        Sets up the error channels from the error rate and error bias input parameters
        '''

        xyz_error_bias = np.array(self.xyz_error_bias)
        if xyz_error_bias[0] == np.inf:
            self.px = self.error_rate
            self.py = 0
            self.pz = 0
        elif xyz_error_bias[1] == np.inf:
            self.px = 0
            self.py = self.error_rate
            self.pz = 0
        elif xyz_error_bias[2] == np.inf:
            self.px = 0
            self.py = 0
            self.pz = self.error_rate
        else:
            self.px, self.py, self.pz = self.error_rate * \
                xyz_error_bias/np.sum(xyz_error_bias)

        if self.hadamard_rotate==0:
            self.channel_probs_x = np.ones(self.N)*(self.px)
            self.channel_probs_z = np.ones(self.N)*(self.pz)
            self.channel_probs_y = np.ones(self.N)*(self.py)
        
        elif self.hadamard_rotate==1:
            n1=self.hadamard_rotate_sector1_length
            self.channel_probs_x =np.hstack([np.ones(n1)*(self.px),np.ones(self.N-n1)*(self.pz)])
            self.channel_probs_z =np.hstack([np.ones(n1)*(self.pz),np.ones(self.N-n1)*(self.px)])
            self.channel_probs_y = np.ones(self.N)*(self.py)
        else:
            raise ValueError(f"The hadamard rotate attribute should be set to 0 or 1. Not '{self.hadamard_rotate}")

        self.channel_probs_x.setflags(write=False)
        self.channel_probs_y.setflags(write=False)
        self.channel_probs_z.setflags(write=False)

    def _decoder_setup(self):

        '''
        Setup for the BP+OSD decoders 
        '''

        # decoder for Z errors
        self.bpd_z = bposd_decoder(
            self.hx,
            channel_probs=self.channel_probs_z+self.channel_probs_y,
            max_iter=self.max_iter,
            bp_method=self.bp_method,
            ms_scaling_factor=self.ms_scaling_factor,
            osd_method=self.osd_method,
            osd_order=self.osd_order,
        )

        # decoder for X-errors
        self.bpd_x = bposd_decoder(
            self.hz,
            channel_probs=self.channel_probs_x+self.channel_probs_y,
            max_iter=self.max_iter,
            bp_method=self.bp_method,
            ms_scaling_factor=self.ms_scaling_factor,
            osd_method=self.osd_method,
            osd_order=self.osd_order,
        )

    def _generate_error(self):

        '''
        Generates a random error on both the X and Z components of the code
        distributed according to the channel probability vectors.
        '''

        for i in range(self.N):
            rand = np.random.random()
            if rand < self.channel_probs_z[i]:
                self.error_z[i] = 1
                self.error_x[i] = 0
            elif self.channel_probs_z[i] <= rand < (self.channel_probs_z[i]+self.channel_probs_x[i]):
                self.error_z[i] = 0
                self.error_x[i] = 1
            elif (self.channel_probs_z[i]+self.channel_probs_x[i]) <= rand < (self.channel_probs_x[i]+self.channel_probs_y[i]+self.channel_probs_z[i]):
                self.error_z[i] = 1
                self.error_x[i] = 1
            else:
                self.error_z[i] = 0
                self.error_x[i] = 0

        return self.error_x, self.error_z
 
    def run_decode_sim(self):

        '''
        This function contains the main simulation loop and controls the output.
        '''

        # save start date
        self.start_date = datetime.datetime.fromtimestamp(
            time.time()).strftime("%A, %B %d, %Y %H:%M:%S")

        pbar = tqdm(range(self.run_count+1, self.target_runs+1),
                    disable=self.tqdm_disable, ncols=0)

        start_time = time.time()
        save_time = start_time

        for self.run_count in pbar:

            self._single_run()

            pbar.set_description(f"d_max: {self.min_logical_weight}; OSDW_WER: {self.osdw_word_error_rate*100:.3g}±{self.osdw_word_error_rate_eb*100:.2g}%; OSDW: {self.osdw_logical_error_rate*100:.3g}±{self.osdw_logical_error_rate_eb*100:.2g}%; OSD0: {self.osd0_logical_error_rate*100:.3g}±{self.osd0_logical_error_rate_eb*100:.2g}%;")

            current_time = time.time()
            save_loop = current_time-save_time

            if int(save_loop)>self.save_interval or self.run_count==self.target_runs:
                save_time=time.time()
                self.runtime = save_loop +self.runtime

                self.runtime_readable=time.strftime('%H:%M:%S', time.gmtime(self.runtime))


                if self.output_file!=None:
                    f=open(self.output_file,"w+")
                    print(self.output_dict(),file=f)
                    f.close()

                if self.osdw_logical_error_rate_eb>0 and self.osdw_logical_error_rate_eb/self.osdw_logical_error_rate < self.error_bar_precision_cutoff:
                    print("\nTarget error bar precision reached. Stopping simulation...")
                    break

        return json.dumps(self.output_dict(),sort_keys=True, indent=4)

    def output_dict(self):

        '''
        Function for formatting the output
        '''

        output_dict = {}
        for key, value in self.__dict__.items():
            if key in self.output_keys:
                output_dict[key] = value
        
        return output_dict
    


def HGP(H1: sp.csr_array, H2: sp.csr_array=None):
    # Convention: H1 is the vertical axis, H2 is the horizontal axis
    # BB | BC (Z stab)
    # CB | CC
    # (X stab)
    if H2 is None:
        H2 = H1
    H1 = H1.astype(np.uint)
    H2 = H2.astype(np.uint)
    (m1, n1), (m2, n2) = H1.shape, H2.shape
    I = lambda n: sp.eye_array(n, dtype=np.uint)
    Hz = sp.hstack([sp.kron(I(n1), H2), sp.kron(H1.T, I(m2))]).asformat('csr')
    Hx = sp.hstack([sp.kron(H1, I(n2)), sp.kron(I(m1), H2.T)]).asformat('csr')
    return Hx, Hz




import h5py
import argparse

import pickle

names = ["PEG_codes", "SA_codes", "PS_codes", "PE_codes"]
objs = []
for name in names:
#     with open(name+'.pkl', 'wb') as f:
#         pickle.dump(obj, f)
    with open(name+'.pkl', 'rb') as f:
        objs.append(pickle.load(f))

codes = ['[625,25]', '[1600,64]', '[2025,81]']

error_rates = [np.logspace(-2, -1, 20), 
               np.logspace(-2, -1, 20), 
               np.logspace(-2, -1, 20)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-F', action="store", dest='F', default=0, type=int, required=True)
    parser.add_argument('-C', action="store", dest='C', default=0, type=int, required=True)
    parser.add_argument('-E', action="store", dest='E', default=0, type=int, required=True)
    parser.add_argument('-M', action="store", dest='MC', default=int(1e4), type=int)
    args = parser.parse_args()

    # Choose the code family, code length, error rate, MC budget
    F, C, E, MC = args.F, args.C, args.E, args.MC
    # Increase the budget for small codes
    if C == 0:
        MC *= 10

    family = objs[F]
    code = list(family.keys())[C]
    er = error_rates[C][E]

    # Define CSS code via HGP construction
    Hx, Hz = HGP(family[code])
    Hx, Hz = Hx.todense(), Hz.todense()

    print(f'Code params: {code} | Family: {names[F]}')
    print(f'Error rate: 10^{np.log10(er):.2f} | MC budget: 10^{np.log10(MC):.0f} trials')

    # Set BP+OSD params and run simulations
    params = {
        "error_rate": er, #the physical error rate on the qubits
        "target_runs": MC, #the number of cycles to simulate
        'max_iter': int(Hx.shape[1]/10), #the interation depth for BP
        'tqdm_disable': True, #show live stats
        'xyz_error_bias': [1, 0, 0], #show live stats
        'channel_update': None,
    }
    simulation = css_decode_sim(hx=Hx, hz=Hz, **params).output_dict()

    # Collect results
    ler, ler_eb = simulation["osdw_logical_error_rate"], simulation["osdw_logical_error_rate_eb"]
    bp_ler, bp_ler_eb = simulation["bp_logical_error_rate"], simulation["bp_logical_error_rate_eb"]
    bp_x_conv_rate = simulation["bp_converge_count_x"]/simulation["run_count"]
    bp_z_conv_rate = simulation["bp_converge_count_z"]/simulation["run_count"]
    
    # Save results
    ler = np.array([ler], dtype=float)
    ler_eb = np.array([ler_eb], dtype=float)
    bp_ler = np.array([bp_ler], dtype=float)
    bp_ler_eb = np.array([bp_ler_eb], dtype=float)
    bp_x_conv_rate = np.array([bp_x_conv_rate], dtype=float)
    bp_z_conv_rate = np.array([bp_z_conv_rate], dtype=float)

    time.sleep(E)
    with h5py.File("bposd_simulations.hdf5", "a") as f: 
        grp = f.require_group(names[F])
        subgrp = grp.require_group(code)
        subsubgrp = subgrp.require_group(f'ER={E}')
        
        if 'ler' in subsubgrp:
            del subsubgrp['ler']
        subsubgrp.create_dataset("ler", data=ler)
        
        if 'ler_eb' in subsubgrp:
            del subsubgrp['ler_eb']
        subsubgrp.create_dataset("ler_eb", data=ler_eb)
        
        if 'bp_ler' in subsubgrp:
            del subsubgrp['bp_ler']
        subsubgrp.create_dataset("bp_ler", data=bp_ler)
        
        if 'bp_ler_eb' in subsubgrp:
            del subsubgrp['bp_ler_eb']
        subsubgrp.create_dataset("bp_ler_eb", data=bp_ler_eb)
        
        if 'bp_x_conv_rate' in subsubgrp:
            del subsubgrp['bp_x_conv_rate']
        subsubgrp.create_dataset("bp_x_conv_rate", data=bp_x_conv_rate)
        
        if 'bp_z_conv_rate' in subsubgrp:
            del subsubgrp['bp_z_conv_rate']
        subsubgrp.create_dataset("bp_z_conv_rate", data=bp_z_conv_rate)
    