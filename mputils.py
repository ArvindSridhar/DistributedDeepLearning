from multiprocessing import Process, Pipe

def spawn(f):
	def fun(pipe,x):
		pipe.send(f(x))
		pipe.close()
	return fun

def parmap(f,X):
	pipe=[Pipe() for x in X]
	proc=[Process(target=spawn(f),args=(c,x)) for x,(p,c) in zip(X,pipe)]
	[p.start() for p in proc]
	[p.join() for p in proc]
	return [p.recv() for (p,c) in pipe]
