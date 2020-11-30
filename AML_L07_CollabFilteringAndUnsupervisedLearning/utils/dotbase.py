from torch import tensor, nn
import torch 


def init_params(size): return nn.Parameter(torch.zeros(*size).normal_(0, 0.01))

def update_params(params, lr):
    params.data -= lr * params.grad.data
    params.grad = None

class DotBase():
    
    def __init__(self, users_shape, movies_shape, X, use_bias=False, lr=.1):
    
        self.use_bias = use_bias
        self.x = X
        self.lr = lr
        
        try:
            self.load(users_shape, movies_shape)
        except FileNotFoundError as e:
            print('Not found')
            self.lat_users = init_params(users_shape)
            self.lat_movies = init_params(movies_shape)
            self.bias_users = init_params((users_shape[0], ))
            self.bias_movies = init_params((movies_shape[0], ))
        
    
    def update_lat(self, bs=64):
        """
        Update the latend factors using learning_rate lr.
        """
        lr = self.lr
        for i in range(int(len(self.x)/bs)):
            # Perform sgd using batch size bs 
            
            loss = self.get_loss(self.x[i*bs: (i+1)*bs, :])
            loss.backward()

            update_params(self.lat_users, lr)
            update_params(self.lat_movies, lr)
            if self.use_bias:
                update_params(self.bias_users, lr)
                update_params(self.bias_movies, lr)
        
        return loss
    
    def get_params(self, idxs, key='user'):
        if key=='user':
            lat = self.lat_users[idxs]
            bias = self.bias_users[idxs]
        elif key=='movie':
            lat = self.lat_movies[idxs]
            bias = self.bias_movies[idxs]
        
        if self.use_bias: return lat, bias
        else: return lat
    
    def save(self):
        torch.save(self.lat_users, 'params/lat_users_{}_{}_bias={}.to'.format(*self.lat_users.shape, self.use_bias))
        torch.save(self.lat_movies, 'params/lat_movies_{}_{}_bias={}.to'.format(*self.lat_movies.shape, self.use_bias))
        torch.save(self.bias_users, 'params/bias_users_{}_{}_bias={}.to'.format(*self.lat_users.shape, self.use_bias))
        torch.save(self.bias_movies, 'params/bias_movies_{}_{}_bias={}.to'.format(*self.lat_movies.shape, self.use_bias))

    def load(self, users_shape, movies_shape):
        self.lat_users = torch.load('params/lat_users_{}_{}_bias={}.to'.format(*users_shape, self.use_bias))
        self.lat_movies = torch.load('params/lat_movies_{}_{}_bias={}.to'.format(*movies_shape, self.use_bias))
        self.bias_users = torch.load('params/bias_users_{}_{}_bias={}.to'.format(*users_shape, self.use_bias))
        self.bias_movies = torch.load('params/bias_movies_{}_{}_bias={}.to'.format(*movies_shape, self.use_bias))

