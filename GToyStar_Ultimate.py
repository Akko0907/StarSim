import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time


class Gstar():
	def __init__(self, N):
		self.__r = np.random.randn(N,2)
		self.__v = np.zeros((N,2))
		self.__acel = np.zeros((N,2))
		self.__rho = np.zeros((N,1))
		self.__P = self.__rho.copy()
		self.N  = N      # numero de particulas
		self.dt = 0.04   # timestep
		self.M  = 2      # massa da estrela
		self.R  = 0.75      # raio final da estrela
		self.h  = 0.04/np.sqrt(self.N/1000)    # smoothing length
		self.k  = 0.1    # equation of state constant
		self.n  = 1      # polytropic index
		self.nu = 1      # damping
		self.m  = self.M/self.N    # massa de uma particula
	
	def __str__(self):
		dic = {'N': self.N, 'dt': self.dt, 'M': self.M, 'R': self.R,
			   'h': self.h,'k': self.k,'n': self.n, 'nu': self.nu,
			   'm': self.m}
		return f'{dic}'
	
	def G(self, x, y):
		# Gausssian Smoothing kernel (2D)
		# x     is a vector/matrix of x positions
		# y     is a vector/matrix of y positions
		# h     is the smoothing length
	
		l = np.sqrt(x**2 + y**2)
		
		return (1.0 / (self.h*np.sqrt(np.pi)))**2 * np.exp( -l**2 / self.h**2)
		
	def gradG(self, x, y):
		# Gradient of the Gausssian Smoothing kernel (2D)
		# x     is a vector/matrix of x positions
		# y     is a vector/matrix of y positions
		# h     is the smoothing length
		# dGx, dGy   is the evaluated gradient
		l = np.sqrt(x**2 + y**2 )
		n = -2 * np.exp( -l**2 / self.h**2)/(self.h**4 * (np.pi))

		dGx = n * x
		dGy = n * y
		return dGx, dGy
		
	def PairInt(self):
		# Organiza a forma das matrizes rx e ry, que conteem,
		# respectivamente,as coordenadas x e y de cada vetor'''
		N = self.__r.shape[0]
		
		rx = self.__r[:,0].reshape((N,1))
		ry = self.__r[:,1].reshape((N,1))	
		
		# Definindo a matriz diferenca ri-rj
		# por meio de dx e dy
		dx = rx - rx.T
		dy = ry - ry.T
		
		return dx, dy
	
	def Up_rho(self):
		# Atualizamos a densidade das particulas por meio da formula
		# rho_i=sum_j(mj*W(ri-rj,h)) ---> retorna rho como matriz Mx1
		
		N = self.__r.shape[0]
		dx, dy = self.PairInt()
		W = self.G(dx,dy)
		
		self.__rho = np.sum(self.m*W,1).reshape((N,1))
		
	def Up_P(self):
		# Calculamos a pressao em cada particula 
		# ---> retorna a pressao como matriz Mx1
		self.__P = self.k*self.__rho**(1+1/self.n)
	  
	def Up_Acel(self, lambd):
		# Calculamos cada componente da aceleracao separadamente,
		# por simplicidade, fazendo a mesma soma descrita em Up_rho
		# e entao juntamos os vetores coluna ax e ay para conseguirmos
		# a aceleracao devida a pressao (np.hstack)
		
		N = self.__r.shape[0]
		dx, dy = self.PairInt()
		dGx, dGy = self.gradG(dx, dy)
		
		self.Up_rho()
		self.Up_P()
		
		ax = np.sum(-self.m*(self.__P/self.__rho**2 + self.__P.T/self.__rho.T**2)*dGx,1).reshape((N,1))
		ay = np.sum(-self.m*(self.__P/self.__rho**2 + self.__P.T/self.__rho.T**2)*dGy,1).reshape((N,1))
		
		a = np.hstack((ax,ay)) # Pressao
		a += -self.nu*self.__v # viscosidade
		a += -lambd*self.__r # Forca externa
		
		self.__acel = a
		
	def Up_rp(self, i):
		lambd  =  (2*self.k*np.pi**(-1/self.n)*(self.M*(1+self.n)/self.R**2)**(1+(1/self.n)))/self.M
		
		# Definimos a aceleracao inicial do processo q eh usada
		# no calculo de v(-dt/2), para entao usufruirmos 
		# do metodo leap frog
		self.Up_Acel(lambd)
		v_mhalf = -self.__acel*self.dt/2
		
		# Esta funcao ao ser chamada descobre as novas posicoes
		# e velocidades das particulas apos i*dt segundos
		for i in range(0,i):
			v_phalf = v_mhalf + self.__acel*self.dt
			self.__r += v_phalf*self.dt
			self.__v = 0.5*(v_mhalf+v_phalf)
			v_mhalf = v_phalf
			
			self.Up_Acel(lambd)
			 
	def Run(self):
		fig, ax = plt.subplots()
		plt.xlim(-2,2)
		plt.ylim(-2,2)
		c = np.random.random((self.N,1)).flatten()
		particles = ax.scatter(self.__r[:,0],self.__r[:,1],c=c,s=10, cmap='YlOrBr') #cmap mapeia cores de acordo com um array determinado em c
		
		def animate(i):
			#perform animation steps
			nonlocal ax,fig
			self.Up_rp(i)
			r = self.__r
			rho = self.__rho
			cvalue = np.minimum(rho/2,1).flatten() #set o array para cor de cada particula
			#update pieces of animation
			particles.set_offsets(r[:,:2]) #set data
			particles.set_array(cvalue) #set color
			return particles,
			
		init = time()
		
		anim = animation.FuncAnimation(fig,animate,60,interval=10,blit=True)
		
		writergif = animation.PillowWriter(fps=60)
		anim.save('GCstartoy(3).gif',writer=writergif)
		
		tempo = time() - init  
		print(f'Levados {round(tempo,2)}s para conclusao do programa\n dados:{self}')
		
		
star = Gstar(1000)

star.Run()