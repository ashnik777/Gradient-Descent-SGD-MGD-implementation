def loss(X,y,w):
    return np.mean((X.dot(w)-y)**2)


def gradient_descent(epoch,X,y):
  alpha = 1e-3
  eps = 1e6
  w = np.random.uniform(-1,1,X.shape[1])
  w_cur = w
  i = 0
  while (i<epoch) and (eps>1e-8):
    if i%100 == 0:
      print("Iteration: {}, Loss: {}".format(i,loss(X,y,w_cur)))
    w_new = w_cur-alpha*(2*(X.T.dot(X).dot(w_cur)-X.T.dot(y)))
    eps = np.linalg.norm(w_new - w_cur)
    w_cur = w_new
    i += 1
  w = w_cur
  loss(X,y,w)
