#If you're reading this file you're looking too far into the code of whatever application you're reading.

"""
 tileGraphics.py
 pygame wrapper simple functions for faster application development
"""

__author__ = "lganic"

import os
import pygame
import numpy as np
from PIL import Image as image
import time,threading

os_name = os.name

if os_name == 'nt':
	import win32gui
	import win32con



keycodes={'BACKSPACE':pygame.K_BACKSPACE,'TAB':pygame.K_TAB,'CLEAR':pygame.K_CLEAR,'RETURN':pygame.K_RETURN,'PAUSE':pygame.K_PAUSE,'ESCAPE':pygame.K_ESCAPE,'SPACE':pygame.K_SPACE,'EXCLAIM':pygame.K_EXCLAIM,'QUOTEDBL':pygame.K_QUOTEDBL,'HASH':pygame.K_HASH,'DOLLAR':pygame.K_DOLLAR,'AMPERSAND':pygame.K_AMPERSAND,'QUOTE':pygame.K_QUOTE,'LEFTPAREN':pygame.K_LEFTPAREN,'RIGHTPAREN':pygame.K_RIGHTPAREN,'ASTERISK':pygame.K_ASTERISK,'PLUS':pygame.K_PLUS,'COMMA':pygame.K_COMMA,'MINUS':pygame.K_MINUS,'PERIOD':pygame.K_PERIOD,'SLASH':pygame.K_SLASH,'0':pygame.K_0,'1':pygame.K_1,'2':pygame.K_2,'3':pygame.K_3,'4':pygame.K_4,'5':pygame.K_5,'6':pygame.K_6,'7':pygame.K_7,'8':pygame.K_8,'9':pygame.K_9,'COLON':pygame.K_COLON,'SEMICOLON':pygame.K_SEMICOLON,'LESS':pygame.K_LESS,'EQUALS':pygame.K_EQUALS,'GREATER':pygame.K_GREATER,'QUESTION':pygame.K_QUESTION,'AT':pygame.K_AT,'LEFTBRACKET':pygame.K_LEFTBRACKET,'BACKSLASH':pygame.K_BACKSLASH,'RIGHTBRACKET':pygame.K_RIGHTBRACKET,'CARET':pygame.K_CARET,'UNDERSCORE':pygame.K_UNDERSCORE,'BACKQUOTE':pygame.K_BACKQUOTE,'a':pygame.K_a,'b':pygame.K_b,'c':pygame.K_c,'d':pygame.K_d,'e':pygame.K_e,'f':pygame.K_f,'g':pygame.K_g,'h':pygame.K_h,'i':pygame.K_i,'j':pygame.K_j,'k':pygame.K_k,'l':pygame.K_l,'m':pygame.K_m,'n':pygame.K_n,'o':pygame.K_o,'p':pygame.K_p,'q':pygame.K_q,'r':pygame.K_r,'s':pygame.K_s,'t':pygame.K_t,'u':pygame.K_u,'v':pygame.K_v,'w':pygame.K_w,'x':pygame.K_x,'y':pygame.K_y,'z':pygame.K_z,'DELETE':pygame.K_DELETE,'KP0':pygame.K_KP0,'KP1':pygame.K_KP1,'KP2':pygame.K_KP2,'KP3':pygame.K_KP3,'KP4':pygame.K_KP4,'KP5':pygame.K_KP5,'KP6':pygame.K_KP6,'KP7':pygame.K_KP7,'KP8':pygame.K_KP8,'KP9':pygame.K_KP9,'KP_PERIOD':pygame.K_KP_PERIOD,'KP_DIVIDE':pygame.K_KP_DIVIDE,'KP_MULTIPLY':pygame.K_KP_MULTIPLY,'KP_MINUS':pygame.K_KP_MINUS,'KP_PLUS':pygame.K_KP_PLUS,'KP_ENTER':pygame.K_KP_ENTER,'KP_EQUALS':pygame.K_KP_EQUALS,'UP':pygame.K_UP,'DOWN':pygame.K_DOWN,'RIGHT':pygame.K_RIGHT,'LEFT':pygame.K_LEFT,'INSERT':pygame.K_INSERT,'HOME':pygame.K_HOME,'END':pygame.K_END,'PAGEUP':pygame.K_PAGEUP,'PAGEDOWN':pygame.K_PAGEDOWN,'F1':pygame.K_F1,'F2':pygame.K_F2,'F3':pygame.K_F3,'F4':pygame.K_F4,'F5':pygame.K_F5,'F6':pygame.K_F6,'F7':pygame.K_F7,'F8':pygame.K_F8,'F9':pygame.K_F9,'F10':pygame.K_F10,'F11':pygame.K_F11,'F12':pygame.K_F12,'F13':pygame.K_F13,'F14':pygame.K_F14,'F15':pygame.K_F15,'NUMLOCK':pygame.K_NUMLOCK,'CAPSLOCK':pygame.K_CAPSLOCK,'SCROLLOCK':pygame.K_SCROLLOCK,'RSHIFT':pygame.K_RSHIFT,'LSHIFT':pygame.K_LSHIFT,'RCTRL':pygame.K_RCTRL,'LCTRL':pygame.K_LCTRL,'RALT':pygame.K_RALT,'LALT':pygame.K_LALT,'RMETA':pygame.K_RMETA,'LMETA':pygame.K_LMETA,'LSUPER':pygame.K_LSUPER,'RSUPER':pygame.K_RSUPER,'MODE':pygame.K_MODE,'HELP':pygame.K_HELP,'PRINT':pygame.K_PRINT,'SYSREQ':pygame.K_SYSREQ,'BREAK':pygame.K_BREAK,'MENU':pygame.K_MENU,'POWER':pygame.K_POWER,'EURO':pygame.K_EURO}



class sound:
	def __init__(self,fileName, folder =''):
		if folder!="":
			if not folder.endswith("/") and not folder.endswith("\\"):
				folder+="\\"
		self.source=folder+fileName
	def play(self):
		pygame.mixer.music.load(self.source)
		pygame.mixer.music.play()




class sizeSprite:
	def __init__(self,graphicsInstance,fileName,width,height,backgroundColor=(-1,-1,-1),folder=""):
		if folder!="":
			if not folder.endswith("/") and not folder.endswith("\\"):
				folder+="\\"
		self.source=folder+fileName
		self.width=width
		self.size=graphicsInstance.tileWidth
		self.height=height
		self.trackerNum=graphicsInstance.spriteReloadNum
		self.backColor=backgroundColor
		self.pixdata=[]
		width=int(width*self.size)
		height=int(height*self.size)
		im=image.open(self.source).resize((width,height)).convert("RGBA")
		for y in range(height):
			buffer=[]
			colorBuffer=[]
			blanks=0
			for x in range(width):
				color=im.getpixel((x,y))[:3][::-1]
				if color!=self.backColor:
					if blanks>0:
						buffer.append(blanks)
						blanks=0
					colorBuffer+=[a for a in color]+[0]
				else:
					if len(colorBuffer)>0:
						buffer.append(colorBuffer.copy())
						colorBuffer=[]
					blanks+=1
			if len(colorBuffer)>0:
				buffer.append(colorBuffer.copy())
				colorBuffer=[]
			self.pixdata.append(buffer.copy())
	def setBackgroundColor(self,backgroundColor):
		self.backColor=backgroundColor
		self.reload()
	def reload(self,graphicsInstance=None):
		if graphicsInstance!=None:
			self.trackerNum=graphicsInstance.spriteReloadNum
			self.size=graphicsInstance.tileWidth
		self.pixdata=[]
		width=int(self.width*self.size)
		height=int(self.height*self.size)
		im=image.open(self.source).resize((width,height)).convert("RGB")
		for y in range(height):
			buffer=[]
			colorBuffer=[]
			blanks=0
			for x in range(width):
				color=im.getpixel((x,y))[:3][::-1]
				if color!=self.backColor:
					if blanks>0:
						buffer.append(blanks)
						blanks=0
					colorBuffer+=[a for a in color]+[0]
				else:
					if len(colorBuffer)>0:
						buffer.append(tuple(colorBuffer.copy()))
						colorBuffer=[]
					blanks+=1
			if len(colorBuffer)>0:
				buffer.append(tuple(colorBuffer.copy()))
				colorBuffer=[]
			self.pixdata.append(tuple(buffer.copy()))

class spriteManager:
	def __init__(self,graphicsInstance,*spriteFolder,backgroundColor=(-1,-1,-1)):
		if len(spriteFolder)==0:
			folder=""
		else:
			folder=spriteFolder[0]
		import os
		self.data={}
		lst=os.listdir(folder if folder!="" else None)
		for a in lst:
			if a.endswith(".png"):
				self.data[a]=sprite(graphicsInstance,a,backgroundColor=backgroundColor,folder=folder)
	def get(self,name):
		if not name.endswith(".png"):
			name+=".png"
		return self.data[name]
	def getNames(self):
		return list(self.data.keys())

class complexSpriteManager:
	def __init__(self,graphicsInstance,folder,backgroundColor=(-1,-1,-1)):
		import os
		self.data={}
		lst=os.listdir(folder)
		for a in lst:
			if a.endswith(".cplx"):
				self.data[a]=complexSprite(graphicsInstance,folder,backgroundColor=backgroundColor,name=a)
	def get(self,name):
		if not name.endswith(".cplx"):
			name+=".cplx"
		return self.data[name]
	def getNames(self):
		return list(self.data.keys())


class colorPalette:
	def __init__(self,*args):
		self.colors=[]
		for a in args:
			self.colors.append(a)
	def get(self,index):
		if type(index)==str:
			return tuple(int(index[i:i+2], 16) for i in (0, 2, 4))#thanks to John1024 for this beautiful line of code
		if type(index)==tuple:
			return index
		return self.colors[index]
	def add(self,color):
		self.colors.append(color)

class legacySprite:
	def __init__(self,graphicsInstance,fileName,backgroundColor=(-1,-1,-1),folder=""):
		if folder!="":
			if not folder.endswith("/") and not folder.endswith("\\"):
				folder+="\\"
		im=image.open(folder+fileName)
		rgb_im =im.convert("RGB")
		self.size=graphicsInstance.tileWidth
		self.pixdata=[]
		import copy
		dy=im.size[1]/self.size
		dx=im.size[0]/self.size
		y=0
		while y<im.size[1]:
			temp=[]
			x=0
			while x<im.size[0]:
				temp.append(rgb_im.getpixel((int(x),int(y))))
				x+=dx
			y+=dy
			self.pixdata.append(copy.copy(temp))
		self.backColor=backgroundColor
	def setBackgroundColor(self,color):
		self.backColor=color

class sprite:
	def __init__(self,graphicsInstance,fileName,backgroundColor=(-1,-1,-1),folder=""):
		if folder!="":
			if not folder.endswith("/") and not folder.endswith("\\"):
				folder+="\\"
		self.source=folder+fileName
		self.size=graphicsInstance.tileWidth
		self.trackerNum=graphicsInstance.spriteReloadNum#used for manual and automatic sprite reloading
		self.pixdata=[]
		width, height = self.size, self.size
		self.backColor=backgroundColor
		if backgroundColor != (-1, -1, -1):
			im=image.open(self.source).convert("RGBA")
			im.putalpha(255) # Nuke alpha
			dat = np.array(im)
			tr, tg, tb = backgroundColor
			r, g, b, a = dat.T
			remFlags = (r==tr) & (g==tg) & (b==tb)
			dat[...,][remFlags.T] = (0,0,0,0)
			im = image.fromarray(dat)
			im = im.resize((width, height))
		else:
			im=image.open(self.source).resize((width,height)).convert("RGBA")
		for y in range(self.size):
			buffer=[]
			colorBuffer=[]
			blanks=0
			for x in range(self.size):
				color=im.getpixel((x,y))[::-1]
				if color[0] == 255:
					if blanks>0:
						buffer.append(blanks)
						blanks=0
					colorBuffer+=[a for a in color[1:]]+[0]
				else:
					if len(colorBuffer)>0:
						buffer.append(colorBuffer.copy())
						colorBuffer=[]
					blanks+=1
			if len(colorBuffer)>0:
				buffer.append(colorBuffer.copy())
				colorBuffer=[]
			self.pixdata.append(buffer.copy())
	def setBackgroundColor(self,color):
		self.backColor=color
		self.reload()
	def reload(self,graphicsInstance=None):
		self.pixdata = []
		if graphicsInstance!=None:
			self.size=graphicsInstance.tileWidth
			self.trackerNum=graphicsInstance.spriteReloadNum
		width, height = self.size, self.size
		if self.backColor != (-1, -1, -1):
			im=image.open(self.source).convert("RGBA")
			im.putalpha(255) # Nuke alpha
			dat = np.array(im)
			tr, tg, tb = self.backColor
			r, g, b, a = dat.T
			remFlags = (r==tr) & (g==tg) & (b==tb)
			dat[...,][remFlags.T] = (0,0,0,0)
			im = image.fromarray(dat)
			im = im.resize((width, height))
		else:
			im=image.open(self.source).resize((width,height)).convert("RGBA")
		for y in range(self.size):
			buffer=[]
			colorBuffer=[]
			blanks=0
			for x in range(self.size):
				color=im.getpixel((x,y))[::-1]
				if color[0] == 255:
					if blanks>0:
						buffer.append(blanks)
						blanks=0
					colorBuffer+=[a for a in color[1:]]+[0]
				else:
					if len(colorBuffer)>0:
						buffer.append(colorBuffer.copy())
						colorBuffer=[]
					blanks+=1
			if len(colorBuffer)>0:
				buffer.append(colorBuffer.copy())
				colorBuffer=[]
			self.pixdata.append(buffer.copy())

class legacyComplexSprite:
	def __init__(self,graphicsInstance,folder,backgroundColor=(-1,-1,-1),name=""):
		if not folder.endswith("/") and not folder.endswith("\\"):
			folder+="\\"
		self.sprites=[]
		import os
		l=os.listdir(folder)
		text="__error__"
		for a in l:
			if a.endswith(".cplx"):
				if name=="" or a==name or a==name+".cplx":
					text=open(folder+a,"r").read()
		l=text.split("\n")
		self.positions=[]
		for a in l:
			temp=a.split(" ")
			self.sprites.append(legacySprite(graphicsInstance,temp[0],backgroundColor=backgroundColor,folder=folder))
			self.positions.append((int(temp[1]),int(temp[2])))


class complexSprite:
	def __init__(self,graphicsInstance,folder,backgroundColor=(-1,-1,-1),name=""):
		self.size=graphicsInstance.tileWidth
		self.backColor=backgroundColor
		self.trackerNum=graphicsInstance.spriteReloadNum
		if not folder.endswith("/") and not folder.endswith("\\"):
			folder+="\\"
		import os
		self.folder=folder
		l=os.listdir(folder)
		text="__error__"
		for a in l:
			if a.endswith(".cplx"):
				if name=="" or a==name or a==name+".cplx":
					text=open(folder+a,"r").read()
					self.source=folder+a
		l=text.split("\n")
		sxp=[]
		syp=[]
		snames=[]
		for a in l:
			temp=a.split(" ")
			snames.append(temp[0])
			sxp.append(int(temp[1]))
			syp.append(int(temp[2]))
		self.positions=zip(sxp,syp)
		self.height=max(syp)+1
		self.width=max(sxp)+1
		mat=[]
		for y in range(self.height):
			mat.append(([None]*self.width).copy())
		for x, y, snames in zip(sxp,syp,snames):
			mat[y][x]=image.open(self.folder+snames).resize((self.size,self.size)).convert("RGB")
		self.pixdata=[]
		for tileY in range(self.height):
			for pixY in range(self.size):
				buffer=[]
				colorBuffer=[]
				blanks=0
				for tileX in range(self.width):
					if mat[tileY][tileX]!=None:
						for pixX in range(self.size):
							color=mat[tileY][tileX].getpixel((pixX,pixY))[:3][::-1]
							if color!=self.backColor:
								if blanks>0:
									buffer.append(blanks)
									blanks=0
								colorBuffer+=[a for a in color]+[0]
							else:
								if len(colorBuffer)>0:
									buffer.append(colorBuffer.copy())
									colorBuffer=[]
								blanks+=1
					else:
						if len(colorBuffer)>0:
							buffer.append(colorBuffer.copy())
							colorBuffer=[]
						blanks+=self.size
				if len(colorBuffer)>0:
					buffer.append(colorBuffer.copy())
				self.pixdata.append(buffer.copy())
	def setBackgroundColor(self,backgroundColor):
		self.backColor=backgroundColor
		self.reload()
	def reload(self,graphicsInstance=None):
		if graphicsInstance!=None:
			self.size=graphicsInstance.tileWidth
			self.trackerNum=graphicsInstance.spriteReloadNum
		text=open(self.source,"r").read()
		l=text.split("\n")
		sxp=[]
		syp=[]
		snames=[]
		for a in l:
			temp=a.split(" ")
			snames.append(temp[0])
			sxp.append(int(temp[1]))
			syp.append(int(temp[2]))
		self.height=max(syp)+1
		self.width=max(sxp)+1
		mat=[]
		for y in range(self.height):
			mat.append(([None]*self.width).copy())
		for x, y, snames in zip(sxp,syp,snames):
			mat[y][x]=image.open(self.folder+snames).resize((self.size,self.size)).convert("RGB")
		self.pixdata=[]
		for tileY in range(self.height):
			for pixY in range(self.size):
				buffer=[]
				colorBuffer=[]
				blanks=0
				for tileX in range(self.width):
					if mat[tileY][tileX]!=None:
						for pixX in range(self.size):
							color=mat[tileY][tileX].getpixel((pixX,pixY))[:3][::-1]
							if color!=self.backColor:
								if blanks>0:
									buffer.append(blanks)
									blanks=0
								colorBuffer+=[a for a in color]+[0]
							else:
								if len(colorBuffer)>0:
									buffer.append(colorBuffer.copy())
									colorBuffer=[]
								blanks+=1
					else:
						if len(colorBuffer)>0:
							buffer.append(colorBuffer.copy())
							colorBuffer=[]
						blanks+=self.size
				if len(colorBuffer)>0:
					buffer.append(colorBuffer.copy())
				self.pixdata.append(buffer.copy())


class textBox:
	def __init__(self,graphicsInstance,x,y,width,height,font="arialms",nRows=5,textColor=(255,255,255),backgroundColor=(0,0,0),border=False,bold=False,rowBoxes=False, center=False):
		self.nRows=nRows
		self.bold=bold
		tw=graphicsInstance.tileWidth
		self.textRect=(tw*x,tw*y,width*tw,height*tw)
		self.border=border
		self.font=font
		self.textColor=textColor
		self.backgroundColor=backgroundColor
		self.x=x
		self.y=y
		self.width=width
		self.height=height
		self.rowBoxes=rowBoxes
		self.center = center
	def setBorder(self,border):
		self.border=border
	def setBackgroundColor(self,color):
		self.backgroundColor=color
	def setTextColor(self,color):
		self.textColor=color
	def setNRows(self,number):
		self.nRows=number
	def setFont(self,font):
		self.font=font
	def setBold(self,bold):
		self.bold=bold

class graphics:
	def __init__(self,width,height,palette="standard",roughSize=-1,screenWidth=None,screenHeight=None,fullScreen=False,roughPercent=-1,fullScreenKey=keycodes["F11"],noResize=False,fps=-1,backgroundImage=None,trackBackgroundLocation=False,showFps=False,alwaysOnTop=False,resizeMode='restrict'):
		if alwaysOnTop and os_name != 'nt':
			print("WARNING: Operating system does not support always-on-top rendering")
			alwaysOnTop = False
		if not noResize and os_name != 'nt' and resizeMode != 'dynamic':
			print("INFO: Forcing resizeMode = dynamic for Linux OS")
			resizeMode = 'dynamic'
		self.alwaysOnTop = alwaysOnTop
		if resizeMode not in ('restrict', 'dynamic'):
			raise ValueError(f'Resize mode invalid: {resizeMode}')
		self.resizeMode = resizeMode
		self.fsKey=fullScreenKey
		self.mouseScroll=0
		self.resized=False
		self.debugKey=None
		self.eventQue=[]
		global pygame,screen,blitScreen
		self.noResize=noResize
		pygame.init()
		pygame.mixer.init()
		pygame.font.init()
		if not '_fScreenSize' in globals():
			obj=pygame.display.Info()
			self.fullScreenWidth=obj.current_w
			self.fullScreenHeight=obj.current_h
			global _fScreenSize
			_fScreenSize=(self.fullScreenWidth,self.fullScreenHeight)
		else:
			self.fullScreenWidth, self.fullScreenHeight=_fScreenSize
		if screenWidth==None:
			screenWidth=self.fullScreenWidth
		if screenHeight==None:
			screenHeight=self.fullScreenHeight
		if roughSize!=-1:
			screenWidth=roughSize
			screenHeight=roughSize
		elif roughPercent!=-1:
			screenWidth=roughPercent*screenWidth
			screenHeight=roughPercent*screenHeight
		if palette=="standard":
			palette=colorPalette((0,0,0),(0,0,255),(0,255,0),(0,255,255),(255,0,0),(255,0,255),(255,255,0),(255,255,255))
		temp1=int(screenWidth/width)
		temp2=int(screenHeight/height)
		self.tileWidth=temp1
		if temp2<temp1:
			self.tileWidth=temp2
		self.width=width
		self.height=height
		self.screenWidth=self.tileWidth*width
		self.screenHeight=self.tileWidth*height
		self.size=(self.screenWidth,self.screenHeight)
		screen=pygame.Surface((self.screenWidth,self.screenHeight))
		if not fullScreen:
			if self.noResize:
				blitScreen=pygame.display.set_mode(self.size)
			else:
				blitScreen=pygame.display.set_mode(self.size,pygame.RESIZABLE)
		else:
			blitScreen=pygame.display.set_mode((0,0),pygame.FULLSCREEN)
			self.size=screen.get_size()
			self.screenWidth=self.size[0]
			self.screenHeight=self.size[1]
		if self.alwaysOnTop:
			hwnd = pygame.display.get_wm_info()["window"]
			win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
		blitScreen.fill(palette.get(0))
		self.matrix=[]
		self.palette=palette
		self.fullScreen=fullScreen
		self.fullscreenToggle=False
		import copy
		temp=[]
		for x in range(width):
			temp.append(0)
		for y in range(height):
			self.matrix.append(copy.copy(temp))
		self.drawCall=True
		self.spriteReloadNum=0
		self.fullScreenOutRes=self.size
		self.resizeBool=False
		self.fullScreenKey=fullScreenKey
		self.targetDelta=1/fps
		self.keyQue=[]
		self.deltaTime=99**99
		self.frametime=time.time()
		self.backgroundMode=False#use matrix data
		if backgroundImage!=None:
			self.setBackgroundImage(backgroundImage)
		self.trackBackgroundLocation=trackBackgroundLocation
		self.showFps=showFps
		if self.showFps:
			self._fpsFont=pygame.font.SysFont("arialms",30,bold=True)
	def setIcon(self,imageName):
		if not "." in imageName:
			imageName+=".png"
		surf=pygame.image.load(imageName)
		pygame.display.set_icon(surf)
	def eventGet(self):
		self.eventQue+=pygame.event.get()
	def update(self,fullQuit=True):
		global blitScreen,screen
		if self.drawCall:
			if self.showFps:
				t=self._fpsFont.render(str(round(1/self.deltaTime,2)),1,(0,255,0))
				tpos=t.get_rect()
				pygame.draw.rect(blitScreen,(0,0,0),tpos)
				screen.blit(t,tpos)
			blitScreen.blit(screen,(0,0))
			pygame.display.flip()
			self.drawCall=False
		events=(self.eventQue+pygame.event.get()).copy()
		self.eventQue=[]
		t=time.time()
		self.deltaTime=t-self.frametime
		self.frametime=t
		if self.deltaTime<self.targetDelta:
			offset=self.targetDelta-self.deltaTime
			time.sleep(offset)
			self.deltaTime+=offset
		for event in events:
			if event.type==pygame.KEYDOWN:
				k=event.key
				if not k in self.keyQue:
					self.keyQue+=[k]
			elif event.type==pygame.KEYUP:
				k=event.key
				if k in self.keyQue:
					self.keyQue.remove(k)
			if event.type==pygame.QUIT:
				pygame.quit()
				if fullQuit:
					exit()
			if event.type == pygame.MOUSEBUTTONDOWN:
				if event.button==4:
					self.mouseScroll+=1
				elif event.button==5:
					self.mouseScroll-=1
			if event.type==pygame.KEYDOWN and event.key==self.fullScreenKey:
				self.fullscreenToggle=True
			if not self.noResize and (event.type==pygame.VIDEORESIZE or self.fullscreenToggle or self.resizeBool):
                #				pygame.image.save(screen,"testscreen.png")
				oldArea=self.width * self.height * self.tileWidth ** 2
				self.resized=True
				#global screen
				if self.fullScreen:
					oldwidthoffset=self.fullScreenWidth-self.width*self.tileWidth
					oldheightoffset=self.fullScreenHeight-self.height*self.tileWidth
				else:
					oldwidthoffset=None
					oldheightoffset=None
				if self.fullscreenToggle:
					self.fullScreen=not self.fullScreen
					if self.fullScreen:
						width=self.fullScreenWidth
						height=self.fullScreenHeight
					else:
						width, height=self.fullScreenOutRes
				elif self.resizeBool:
					width, height=self.size
				else:
					width, height=(event.w,event.h)
				newArea = width * height
				self.fullscreenToggle=False
				oldwidth=self.tileWidth
				temp1=int(width/self.width)
				temp2=int(height/self.height)
				if newArea > oldArea:
					self.tileWidth=max(temp1,temp2)
				else:
					self.tileWidth=min(temp1,temp2)
				self.screenWidth=self.tileWidth*self.width
				self.screenHeight=self.tileWidth*self.height
				self.size=(self.screenWidth,self.screenHeight)
#				if oldwidthoffset==None:
#					screenBackup=pygame.transform.scale(screen.copy(),(self.screenWidth,self.screenHeight))
#				else:
#					screenBackup=pygame.Surface((self.fullScreenWidth-oldwidthoffset,self.fullScreenHeight-oldheightoffset))
#					screenBackup.blit(screen,(0,0),(0,0,self.fullScreenWidth-oldwidthoffset,self.fullScreenHeight-oldheightoffset))
#					screenBackup=pygame.transform.scale(screenBackup,(self.screenWidth,self.screenWidth))
				if self.resizeMode == 'restrict':
					if not self.fullScreen:
						self.fullScreenOutRes=self.size
						blitScreen=pygame.display.set_mode(self.size,pygame.RESIZABLE)
					else:
						blitScreen=pygame.display.set_mode((self.fullScreenWidth,self.fullScreenHeight),pygame.FULLSCREEN)
				if self.alwaysOnTop:
					hwnd = pygame.display.get_wm_info()["window"]
					win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
				screen=pygame.transform.scale(screen,self.size)
				if not self.resizeBool:
					blitScreen.blit(screen,(0,0))
					tempsize=blitScreen.get_size()
					b1=tempsize[0]>self.screenWidth
					b2=tempsize[1]>self.screenHeight
					if b1:
						pygame.draw.rect(blitScreen,self.palette.get(0),(self.screenWidth,0,tempsize[0]-self.screenWidth,self.screenHeight))
					if b2:
						pygame.draw.rect(blitScreen,self.palette.get(0),(0,self.screenHeight,self.screenWidth,tempsize[1]-self.screenHeight))
					if b1 and b2:
						pygame.draw.rect(blitScreen,self.palette.get(0),(self.screenWidth,self.screenHeight,tempsize[0]-self.screenWidth,tempsize[1]-self.screenHeight))
				else:
					screen.fill((0,0,0))
					self.resizeBool=False
					blitScreen.blit(screen,(0,0))
				pygame.display.flip()
				pygame.event.get()
				if oldwidth!=self.tileWidth:
					self.spriteReloadNum+=1
				if self.backgroundMode:
					self.setBackgroundImage(self.backgroundImageSource,width=self.backgroundWidth,height=self.backgroundHeight)
	def setResizeable(self,resizeable):
		global screen
		if resizeable==self.noResize:
			self.noResize=not resizeable
			if self.noResize:
				blitScreen=pygame.display.set_mode((self.screenWidth,self.screenHeight))
			else:
				blitScreen=pygame.display.set_mode((self.screenWidth,self.screenHeight),pygame.RESIZABLE)
			blitScreen.blit(screen,(0,0))
			if self.alwaysOnTop:
				hwnd = pygame.display.get_wm_info()["window"]
				win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
	def setBackgroundImage(self,backgroundImage,width=None,height=None,size=None):
		twidth=self.width
		theight=self.height
		self.backgroundWidth=width
		self.backgroundHeight=height
		if size!=None:
			twidth=size[0]
			theight=size[1]
		elif width!=None and height!=None:
			twidth=width
			theight=height
		twidth*=self.tileWidth
		theight*=self.tileWidth
		self.backgroundImageSource=backgroundImage
		self.backgroundMode=True#use background image
		if not "." in backgroundImage:
			backgroundImage+=".png"
		img=image.open(backgroundImage).resize((twidth,theight))#this is stupid, i need to rewrite all this at some point
		nimg=image.new("RGB",(img.size[0]+2*self.screenWidth,img.size[1]+2*self.screenHeight))
		nimg.paste(img,(self.screenWidth,self.screenHeight))
		self.backgroundSurface=pygame.image.fromstring(nimg.tobytes(),nimg.size,nimg.mode)
	def setBackgroundPosition(self,x,y):
		self.backgroundX=x
		self.backgroundY=y
		self.trackBackgroundLocation=True
	def drawBackground(self,x=0,y=0):
		if self.trackBackgroundLocation:
			if hasattr(self,"backgroundX"):
				x=self.backgroundX
				y=self.backgroundY
			else:
				self.backgroundX=x
				self.backgroundY=y
		self.drawCall=True
		if not self.backgroundMode:
			self.redraw()
		else:
			x+=self.width
			y+=self.height
			x=int(self.tileWidth*x)
			y=int(self.tileWidth*y)
			area=(x,y,self.screenWidth,self.screenHeight)
			screen.blit(self.backgroundSurface,(0,0),area=area)
	def getResizedBool(self):
		out=self.resized
		self.resized=False
		return out
	def setFPS(self,fps):
		self.targetDelta=1/fps
	def resize(self,width,height):
		self.width=width
		self.height=height
		self.resizeBool=True
		self.matrix=[]
		for y in range(height):
			self.matrix.append(([0]*self.width).copy())
		self.update()
	def resizeScreen(self,pixelWidth,pixelHeight):
		self.size=(pixelWidth,pixelHeight)
		self.resizeBool=True
		self.update()
	def screenshot(self,imageName):
		pygame.image.save(screen,imageName)
	def pixelPut(self,x,y,colorIndex):
		self.drawCall=True
		screen.set_at((x,y),self.palette.get(colorIndex))
	def translate(self,x,y,deltaX,deltaY,width=1,height=1):
		self.drawCall=True
		self.pixelTranslate((x*self.tileWidth,y*self.tileWidth,width*self.tileWidth,height*self.tileWidth),deltaX*self.tileWidth,deltaY*self.tileWidth)
	def pixelTranslate(self,rect,deltaX,deltaY):
		self.drawCall=True
		tempSurface=screen.subsurface(rect)
		finalSurface=pygame.Surface((rect[2],rect[3]))
		finalSurface.blit(tempSurface,(0,0))
		screen.blit(finalSurface,(rect[0]+deltaX,rect[1]+deltaY))
	def pixelRect(self,rect,colorIndex,lineWidth=None):
		self.drawCall=True
		if lineWidth==None:
			pygame.draw.rect(screen,self.palette.get(colorIndex),rect)
		else:
			pygame.draw.rect(screen,self.palette.get(colorIndex),rect,lineWidth)
	def pixelLine(self,c1,c2,colorIndex,width=1):
		self.drawCall=True
		pygame.draw.line(screen,self.palette.get(colorIndex),c1,c2,width)
	def line(self,c1,c2,colorIndex,width=1):
		self.drawCall=True
		pygame.draw.line(screen,self.palette.get(colorIndex),(c1[0]*self.tileWidth+int(self.tileWidth/2),c1[1]*self.tileWidth+int(self.tileWidth/2)),(c2[0]*self.tileWidth+int(self.tileWidth/2),c2[1]*self.tileWidth+int(self.tileWidth/2)),width)
	def put(self,x,y,colorIndex,width=1,height=1):
		self.drawCall=True
		for a in range(width):
			for b in range(height):
#				self.matrix[int(y+b)][int(x+a)]=colorIndex
				pygame.draw.rect(screen,self.palette.get(colorIndex),((x+a)*self.tileWidth,(y+b)*self.tileWidth,self.tileWidth,self.tileWidth))
	def get(self,x,y):
		return self.matrix[y][x]
	def redraw(self):
		self.drawCall=True
		for a in range(self.width):
			for b in range(self.height):
				pygame.draw.rect(screen,self.palette.get(self.matrix[b][a]),((a)*self.tileWidth,(b)*self.tileWidth,self.tileWidth,self.tileWidth))
	def setName(self,name):
		self.drawCall=True
		pygame.display.set_caption(name)
	def fill(self,colorIndex):
		self.drawCall=True
		screen.fill(self.palette.get(colorIndex))
		for a in range(self.width):
			for b in range(self.height):
				self.matrix[b][a]=colorIndex
	def mouse(self):
		return pygame.mouse.get_pos()
	def rawMouseTile(self):
		x, y = self.mouse()
		return (x / self.tileWidth, y / self.tileWidth)
	def mouseTile(self):
		x, y = self.rawMouseTile()
		return (int(x), int(y))
	def outline(self,color,x=0,y=0,width=-1,height=-1,thickness=1):
		self.drawCall=True
		color=self.palette.get(color)
		if width==-1:
			width=self.screenWidth
		else:
			width=width*self.tileWidth
		if height==-1:
			height=self.screenHeight
		else:
			height=height*self.tileWidth
		fx=x*self.tileWidth
		fy=y*self.tileWidth
		for x in range(fx,width+1,self.tileWidth):
			pygame.draw.line(screen,color,(x,0),(x,height),thickness)
		for y in range(fy,height+1,self.tileWidth):
			pygame.draw.line(screen,color,(0,y),(width,y),thickness)
	def checkKey(self,key):
		if key==self.fsKey:
			print("TILEGRAPHICS WARNING, the key being checked is internally bound to fullscreen toggle\nIf you want to use this key, change the fullscreen key with the fullScreenKey=<key> when initializing graphics instance")
		return key in self.keyQue
	def checkClick(self,button=0):
		return pygame.mouse.get_pressed()[button]==1
	def quit(self):
		pygame.quit()
	def highlight(self,x,y,color,width=1,height=1,lineWidth=3):
		self.drawCall=True
		pygame.draw.rect(screen,self.palette.get(color),(x*self.tileWidth,y*self.tileWidth,width*self.tileWidth,height*self.tileWidth),lineWidth)
	def reloadSprites(self):
		self.spriteReloadNum+=1
	def putLegacySprite(self,x,y,sprite):
		self.drawCall=True
		xpart=x-int(x)
		ypart=y-int(y)
		x=int(x)
		y=int(y)
		xpart*=self.tileWidth
		ypart*=self.tileWidth
		xpart=int(xpart)
		ypart=int(ypart)
		px=x*self.tileWidth
		py=y*self.tileWidth
		for y in range(self.tileWidth):
			for x in range(self.tileWidth):
				pix=sprite.pixdata[y][x]
				if pix!=sprite.backColor:
					screen.set_at((x+px+xpart,y+py+ypart),pix)
	def msp(self,x,y,sprite):
		import time
		totaltime=0
		mt=time.time()
		if sprite.trackerNum!=self.spriteReloadNum:
			sprite.reload(graphicsInstance=self)
		self.drawCall=True
		x*=self.tileWidth
		y*=self.tileWidth
		x=int(x)
		y=int(y)
		bufferObject=screen.get_buffer()
		buffer=bytearray(bufferObject)
#		ntArray=list(buffer.raw)
		myData=sprite.pixdata[:self.screenHeight-y]
		for yOff, line in enumerate(myData):
			xOff=0
			for item in line:
				if type(item)==int:
					xOff+=item
				else:
					startIndex=(yOff+y)*self.screenWidth*4+(xOff+x)*4
					t=time.time()
					buffer[startIndex:len(item)+startIndex]=bytes(item)
					totaltime+=time.time()-t
#					buffer.write(bytes(item),offset=startIndex)
#					ntArray[startIndex:startIndex+len(item)]=item
					xOff+=int(len(item)/4)
		temp=time.time()
		bufferObject.write(bytes(buffer))
		print(time.time()-temp)
#		buffer.write(bytes(ntArray))
		print(totaltime,time.time()-mt)
		del buffer
	def putSprite(self,x,y,sprite):
		if sprite.trackerNum!=self.spriteReloadNum:
			sprite.reload(graphicsInstance=self)
		self.drawCall=True
		x*=self.tileWidth
		y*=self.tileWidth
		x=int(x)
		y=int(y)
		buffer=screen.get_buffer()
#		ntArray=list(buffer.raw)
		myData=sprite.pixdata[:self.screenHeight-y]
		for yOff, line in enumerate(myData):
			xOff=0
			for item in line:
				if type(item)==int:
					xOff+=item
				else:
					startIndex=(yOff+y)*self.screenWidth*4+(xOff+x)*4
					buffer.write(bytes(item),offset=startIndex)
#					ntArray[startIndex:startIndex+len(item)]=item
					xOff+=int(len(item)/4)
#		buffer.write(bytes(ntArray))
		del buffer
	def putComplexSprite(self,x,y,complexSprite):
		self.putSprite(x,y,complexSprite)
	def putSizeSprite(self,x,y,sizeSprite):
		self.putSprite(x,y,sizeSprite)
	def putTextArray(self,textBox,textArray):
		self.drawCall=True
		if len(textArray)>textBox.nRows:
			raise(IndexError("size of text array is too large for textBox object to handle"))
		pygame.draw.rect(screen,textBox.backgroundColor,(textBox.x*self.tileWidth,textBox.y*self.tileWidth,textBox.width*self.tileWidth,textBox.height*self.tileWidth))
		if textBox.border:
			pygame.draw.rect(screen,textBox.textColor,(textBox.x*self.tileWidth,textBox.y*self.tileWidth,textBox.width*self.tileWidth,textBox.height*self.tileWidth),1)
		textSize=(textBox.height*self.tileWidth)/textBox.nRows
		font=pygame.font.SysFont(textBox.font,int(textSize),bold=textBox.bold)
		y=textBox.y*self.tileWidth
		for a in textArray:
			if textBox.rowBoxes:
				pygame.draw.rect(screen,textBox.textColor,(textBox.x*self.tileWidth,int(y),textBox.width*self.tileWidth,textSize),1)
			t=font.render(a,1,textBox.textColor)
			tpos=t.get_rect()
			tpos[0]=textBox.x*self.tileWidth
			tpos[1]=int(y)
			if textBox.center:
				tpos[0] += (textBox.width * self.tileWidth - tpos[2]) // 2
			screen.blit(t,tpos)
			y+=textSize
	def runInputBox(self,textBox,addText=""):
		self.keyQue=[]
		output=""
		ex=False
		while not ex:
			self.update()
			for a in self.keyQue:
				if a==keycodes["BACKSPACE"]:
					output=output[:-1]
				else:
					if a!=keycodes["RETURN"] and a<0x110000:
						output+=chr(a)
					else:
						ex=True
			self.keyQue=[]
			self.putTextArray(textBox,[addText+output])
		return output
	def putTextBox(self,textBox,text):
		self.drawCall=True
		pygame.draw.rect(screen,textBox.backgroundColor,(textBox.x*self.tileWidth,textBox.y*self.tileWidth,textBox.width*self.tileWidth,textBox.height*self.tileWidth))
		if textBox.border:
			pygame.draw.rect(screen,textBox.textColor,(textBox.x*self.tileWidth,textBox.y*self.tileWidth,textBox.width*self.tileWidth,textBox.height*self.tileWidth),1)
		textSize=textBox.height*self.tileWidth
		font=pygame.font.SysFont(textBox.font,int(textSize),bold=textBox.bold)
		y=textBox.y*self.tileWidth
		t=font.render(textBox,1,textBox.textColor)
		tpos=t.get_rect()
		tpos[0]=textBox.x*self.tileWidth
		tpos[1]=int(y)
		screen.blit(t,tpos)
		y+=textSize
	def putLegacyComplexSprite(self,x,y,complexSprite):
		self.drawCall=True
		for parse in enumerate(complexSprite.sprites):
			nx=complexSprite.positions[parse[0]][0]+x
			ny=complexSprite.positions[parse[0]][1]+y
			self.putLegacySprite(nx,ny,parse[1])
	def putSecLegacyComplexSprite(self,x,y,complexSprite,spritex,spritey):
		self.drawCall=True
		for parse in enumerate(complexSprite.sprites):
			nx=complexSprite.positions[parse[0]][0]
			ny=complexSprite.positions[parse[0]][1]
			if nx==spritex and ny==spritey:
				self.putLegacySprite(nx+x,ny+y,parse[1])
	def removeSprite(self,x,y):
		self.drawCall=True
		if not self.backgroundMode:
			xpart=x-int(x)
			ypart=y-int(y)
			xrange=[0,1]
			if xpart!=0:
				xrange=[0,2]
			yrange=[0,1]
			if ypart!=0:
				yrange=[0,2]
			x=int(x)
			y=int(y)
			for tx in range(xrange[0],xrange[1]):
				for ty in range(yrange[0],yrange[1]):
					x1=x+tx
					y1=y+ty
					if x1<self.width and x1>=0 and y1<self.height and y1>=0:
						pygame.draw.rect(screen,self.palette.get(self.matrix[y1][x1]),((x1)*self.tileWidth,(y1)*self.tileWidth,self.tileWidth,self.tileWidth))
		else:
			sx=int(x*self.tileWidth)
			sy=int(y*self.tileWidth)
			x+=self.width
			y+=self.height
			if self.trackBackgroundLocation and hasattr(self,"backgroundX"):
				x+=self.backgroundX
				y+=self.backgroundY
			x=int(x*self.tileWidth)
			y=int(y*self.tileWidth)
			screen.blit(self.backgroundSurface,(sx,sy),area=(x,y,self.tileWidth,self.tileWidth))
	def removeLegacyComplexSprite(self,x,y,complexSprite):
		self.drawCall=True
		for parse in enumerate(complexSprite.sprites):
			nx=complexSprite.positions[parse[0]][0]+x
			ny=complexSprite.positions[parse[0]][1]+y
			self.removeSprite(nx,ny)
	def removeComplexSprite(self,x,y,complexSprite):
		self.drawCall=True
		for xo, yo in complexSprite.positions:
			self.removeSprite(x+xo,y+yo)
	def removeSizeSprite(self,x,y,sizeSprite):
		import math
		self.drawCall=True
		if not self.backgroundMode:
			for yo in range(math.ceil(sizeSprite.height)):
				for xo in range(math.ceil(sizeSprite.width)):
					self.put(int(x+xo),int(y+yo),self.get(x+xo,y+yo))
		else:
			sx=int(x*self.tileWidth)
			sy=int(y*self.tileWidth)
			x+=self.width
			y+=self.height
			if self.trackBackgroundLocation and hasattr(self,"backgroundX"):
				x+=self.backgroundX
				y+=self.backgroundY
			x=int(x*self.tileWidth)
			y=int(y*self.tileWidth)
			screen.blit(self.backgroundSurface,(sx,sy),area=(x,y,math.ceil(self.tileWidth*sizeSprite.width),math.ceil(self.tileWidth*sizeSprite.height)))
