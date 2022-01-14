from netvlad import NetVLAD
from netvlad import EmbedNet
import sklearn.metrics.pairwise as pairwise
import numpy as np
import torch.nn as nn

import torch
from torch.autograd import Variable
from torch.utils.data import  TensorDataset, DataLoader

cuda = 0

class LSTM(nn.Module):
 
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                    num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
 
        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)
 
        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim*2, output_dim)
 
    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim))
 
    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        input = input.float()
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
        feature_map = (lstm_out[-1].view(self.batch_size, -1))
        feature_map = feature_map.view(feature_map.size(0),feature_map.size(1),1,1)
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))

        return y_pred,feature_map

'''
def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: Variable of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: Variable of shape (batch_size, batch_size)
    """

    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = torch.mm(embeddings, embeddings.t())

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = torch.diag(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = torch.unsqueeze(square_norm, 0) - 2.0 * dot_product + torch.unsqueeze(square_norm, 1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = torch.clamp(distances, min=0.0)

    if not squared:
        # Not sure if needed for pytorch but does not harm
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * 1e-16

        distances = torch.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances

def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: torch.Tensor with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.size(0)).byte()
    if labels.is_cuda:
        indices_equal = indices_equal.cuda()
    indices_not_equal = ~indices_equal
    i_not_equal_j = torch.unsqueeze(indices_not_equal, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_equal, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_equal, 0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    #if labels.is_cuda:
    #    label_equal = label_equal.cuda()
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)

    valid_labels = i_equal_j & (~i_equal_k)
    
    # Combine the two masks
    mask = distinct_indices & valid_labels.type(torch.uint8)

    return mask

def largest_indices(array: np.ndarray, n: int) -> tuple:
    """Returns the n largest indices from a numpy array.
    Arguments:
        array {np.ndarray} -- data array
        n {int} -- number of elements to select
    Returns:
        tuple[np.ndarray, np.ndarray] -- tuple of ndarray
        each ndarray is index
    """
    flat = array.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return ((indices+1)/2+0.5).astype('int64')

def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones.
    Args:
        labels: Variable with labels of the batch, of size (batch_size,)
        embeddings: Variable with tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    anchor_positive_dist = torch.unsqueeze(pairwise_dist, 2)
    anchor_negative_dist = torch.unsqueeze(pairwise_dist, 1)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    mask = Variable(mask.float())
    triplet_loss = mask * triplet_loss

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = torch.clamp(triplet_loss, min=0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = torch.gt(triplet_loss, 1e-16)
    num_positive_triplets = valid_triplets.sum().float()
    num_valid_triplets = mask.sum()
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets

train_total_class = np.load('./models/npy/train_total.npy')
test_total_class = np.load('./models/npy/test_total.npy')
class_cnt = 20

x_data = torch.from_numpy(train_total_class[:,:class_cnt*2,:])
x_test_data = torch.from_numpy(test_total_class[:,:class_cnt,:])
tmp_train = list()
tmp_test = list()
for i in range(class_cnt) :  
  tmp_train.append(i)
  tmp_train.append(i)
  tmp_test.append(i)
y_test_data = np.array(tmp_test)
y_test_data = torch.from_numpy(y_test_data)
y_data = np.array(tmp_train)
y_data = torch.from_numpy(y_data)

dataset_VLAD = TensorDataset(x_data.permute(1,0,2), y_data.type(torch.LongTensor))
loader_VLAD = DataLoader(dataset_VLAD, batch_size=10, shuffle=True)

dataset_VLAD_test = TensorDataset(x_test_data.permute(1,0,2), y_test_data.type(torch.LongTensor))
loader_VLAD_test = DataLoader(dataset_VLAD_test, batch_size=10, shuffle=True)

back_bone_model = LSTM(8, 32, batch_size=10, output_dim=class_cnt, num_layers=2)
back_bone_model.load_state_dict(torch.load('./models/checkpoints/chekcpoint_20200506_best_top3.pth'), strict=True)

# Define model for embedding
net_vlad = NetVLAD(num_clusters=20, dim=64, alpha=1.0)

if (cuda) :
    model = EmbedNet(back_bone_model, net_vlad).cuda()
else :
    model = EmbedNet(back_bone_model, net_vlad)



# criterion = HardTripletLoss(margin=0.1).cuda()
# criterion = batch_all_triplet_loss(margin=0.1).cuda()
optimiser_netvlad = torch.optim.Adam(model.parameters(), lr=0.001)
'''
'''
train_total_class = np.load('./models/npy/train_total.npy')
test_total_class = np.load('./models/npy/test_total.npy')
class_cnt = 20

x_data = torch.from_numpy(train_total_class[:,:class_cnt*2,:])
x_test_data = torch.from_numpy(test_total_class[:,:class_cnt,:])
tmp_train = list()
tmp_test = list()
for i in range(class_cnt) :  
  tmp_train.append(i)
  tmp_train.append(i)
  tmp_test.append(i)
y_test_data = np.array(tmp_test)

y_test_data = torch.from_numpy(y_test_data)
y_data = np.array(tmp_train)
y_data = torch.from_numpy(y_data)
x_data = torch.from_numpy(train_total_class[:,:class_cnt*2,:])
x_test_data = torch.from_numpy(test_total_class[:,:class_cnt,:])

dataset_VLAD_test = TensorDataset(x_test_data.permute(1,0,2), y_test_data.type(torch.LongTensor))
loader_VLAD_test = DataLoader(dataset_VLAD_test, batch_size=10, shuffle=True)
'''

back_bone_model = LSTM(8, 32, batch_size=10, output_dim=class_cnt, num_layers=2)
net_vlad = NetVLAD(num_clusters=20, dim=64, alpha=1.0)

if cuda :
    model = EmbedNet(back_bone_model, net_vlad).cuda()
    model.load_state_dict(torch.load('./models/checkpoints/VLAD_Checkpoint_20200506.pth'), strict=True)

else :
    model = EmbedNet(back_bone_model, net_vlad)
    model.load_state_dict(torch.load('./models/checkpoints/VLAD_Checkpoint_20200506.pth', map_location=torch.device('cpu')), strict=True)

model.eval()

for xx_vlad_test, yy_vlad_test in loader_VLAD_test:
  xx_vlad_test = xx_vlad_test.permute(1,0,2)
  if cuda :
      vald_out_test = model(xx_vlad_test.cuda())
  else :
      vald_out_test = model(xx_vlad_test)


'''
vald_out_test.size()
similarity = pairwise.cosine_similarity(X=vald_out_test.cpu().detach().numpy(), Y=vald_out_train.cpu().detach().numpy(), dense_output=True)

def largest_indices(array: np.ndarray, n: int) -> tuple:
    """Returns the n largest indices from a numpy array.
    Arguments:
        array {np.ndarray} -- data array
        n {int} -- number of elements to select
    Returns:
        tuple[np.ndarray, np.ndarray] -- tuple of ndarray
        each ndarray is index
    """
    flat = array.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return ((indices+1)/2+0.5).astype('int64')

for i in range(len(similarity)) :
  print(largest_indices(similarity[i],3))
'''


print(yy_vlad_test)
