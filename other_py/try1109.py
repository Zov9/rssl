import torch
import time

'''
#how much time it need to calculate two huge matrix matipulation, divide etc,  about 1 to 2 seconds if gpus is empty
# Define the dimensions
dim = 32000
placeholder_tensor = torch.empty(35000, 64000).cuda(2)
placeholder_tensor = None
# Create two large random tensors
tensor1 = torch.randn(dim, 128).cuda(2)
tensor2 = torch.randn(dim, 128).cuda(2)

# Measure the time before the operation
start_time = time.time()

norms_A = torch.norm(tensor1, dim=1, keepdim=True)
norms_B = torch.norm(tensor2, dim=1, keepdim=True)   
# Perform the matrix multiplication
result = torch.mm(tensor1, tensor2.t())
norms_product = torch.mm(norms_A, norms_B.t())
cosine_similarity0 = torch.div(result[:1000],norms_product[:1000])
cosine_similarity1 = torch.div(result[1000:2000],norms_product[1000:2000])
# Measure the time after the operation
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f"Time taken for the operation: {elapsed_time} seconds")

# Example 2D tensor
#tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
'''
'''
# Finding the minimum and maximum values
minimum_value = torch.min(cosine_similarity)
maximum_value = torch.max(cosine_similarity)

print(f"Minimum value: {minimum_value.item()}")
print(f"Maximum value: {maximum_value.item()}")
end_time1 = time.time()
elapsed_time = end_time1 - start_time

print(f"Time taken for the operation: {elapsed_time} seconds")
'''

#Take gpu memory beforehand
'''

# Create a placeholder tensor to occupy memory
placeholder_tensor = torch.empty(35000, 64000).cuda(2)  #9308

# Check the current GPU memory usage
while 1:
    pass
    '''
'''
list_of_tensors = [torch.randn(3) for _ in range(5)]

# stack them along the first dimension
tensor_of_tensors = torch.stack(list_of_tensors, dim=0)

# print the shape of the resulting tensor
print(tensor_of_tensors.shape) # torch.Size([5, n])
print(tensor_of_tensors[0])
'''
def cos_sim():
    a = torch.randn(10)
    maxd = torch.max(a)
    mind = torch.min(a)
    print("maxd,mind",maxd,mind)
    maxd = maxd.item()
    mind  = mind.item()
    print("maxd,mind",maxd,mind)
    return maxd,mind

a = torch.ones(10,10)
fdist = a*-2

ndist = a*-2



for i in range(10):
     for j in range(i,10):
        
        
        fdist[i][j],ndist[i][j] = cos_sim()
        print("fdist[i][j],ndist[i][j]",fdist[i][j],ndist[i][j])
        
