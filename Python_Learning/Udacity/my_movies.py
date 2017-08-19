import movie_class
import movie_website
	
syriana = movie_class.Movie("Syriana", "Yet another CIA mess in a Middle East monarchy", "http://www.impawards.com/2005/posters/syriana_ver2.jpg", "https://www.youtube.com/watch?v=JvTni7Nggi0")

arbitrage = movie_class.Movie("Arbitrage", "Power is the best alibi", "http://www.impawards.com/2012/posters/arbitrage_ver4.jpg", "https://www.youtube.com/watch?	v=UmJSV9ePx7c")

movie_list = [syriana, arbitrage]

movie_website.open_movies_page(movie_list)

