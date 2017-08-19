import turtle

def square_pattern() : 
	window  =turtle.Screen()
	window.bgcolor("red")

	turt = turtle.Turtle()	
	turtle.color("yellow")

	theta = 10
	num_squares = 360/10

	while(num_squares) : 
		#single_square(turt)

		num_reps=4
		while(num_reps) : 
			turt.forward(100)
			turt.right(90)
			num_reps = num_reps-1

		turt.right(theta)
		num_squares = num_squares-1

	window.exitonclick()

square_pattern()

