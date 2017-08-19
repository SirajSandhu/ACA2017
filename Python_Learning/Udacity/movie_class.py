import webbrowser

class Movie() : 
	
	""" defines a class Movie with the title, storyline, poster image and youtube trailer as instance variables """

	def __init__(self, movie_title, storyline, poster_image, youtube_trailer) :
		self.title = movie_title
		self.storyline = storyline
		self.poster = poster_image
		self.trailer = youtube_trailer

	def show_trailer(self) :
		webbrowser.open(self.trailer)
