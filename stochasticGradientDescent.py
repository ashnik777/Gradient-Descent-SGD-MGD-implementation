def loss(X,y,w,b):
    return np.mean((w*X +b -y)**2)


def StochasticGradDescent(X,y):

  w = np.random.uniform(-1,1,1)
  b = np.random.uniform(-1,1,1)
  w_cur = w
  b_cur = b
  eps = 1e-8
  diff = 1e6
  epoch = 15
  alpha = 1e-3
  j=0

  while (j<epoch) and (diff>eps):
    ind = np.arange(X_train.shape[0])
    np.random.shuffle(ind)

    for i in ind:
      w_new = w_cur-alpha*2*(w_cur*X_train[i]+b_cur-y_train[i])*X_train[i]
      b_new = b_cur-alpha*2*(w_cur*X_train[i]+b_cur-y_train[i])
      diff = np.linalg.norm(w_new - w_cur)
      w_cur = w_new
      b_cur = b_new


    print("Epoch: {}, Loss: {}".format(j,loss(X_train,y_train,w_cur,b_cur)))
    j+=1
  return w_cur,b_cur



