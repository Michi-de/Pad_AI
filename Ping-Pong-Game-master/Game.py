# Game.py

import turtle as t
import os

class PongGameAI:
    def __init__(self):
        # Score variable
        self.AI_score = 0

        # Set up the screen
        self.win = t.Screen()
        self.win.title("Ping-Pong Game - AI vs Wall")
        self.win.bgcolor('black')
        self.win.setup(width=800, height=600)
        self.win.tracer(0)

        # Left paddle (AI controlled) - start in the center
        self.paddle_left = t.Turtle()
        self.paddle_left.speed(0)
        self.paddle_left.shape('square')
        self.paddle_left.color('blue')
        self.paddle_left.shapesize(stretch_wid=5, stretch_len=1)
        self.paddle_left.penup()
        self.paddle_left.goto(-350, 0)  # Start in the middle of the left side

        # Ball
        self.ball = t.Turtle()
        self.ball.speed(0)
        self.ball.shape('circle')
        self.ball.color('yellow')
        self.ball.penup()
        self.ball.goto(0, 0)
        self.ball_dx = 5  # Initial speed in x direction
        self.ball_dy = 5  # Initial speed in y direction

        # Score display
        self.pen = t.Turtle()
        self.pen.speed(0)
        self.pen.color('skyblue')
        self.pen.penup()
        self.pen.hideturtle()
        self.pen.goto(0, 260)
        self.update_score()

    def update_score(self):
        """Update the score display at the top of the screen."""
        self.pen.clear()
        self.pen.write(f"AI Score: {self.AI_score}", align="center", font=('Monaco', 24, "normal"))

    def reset_ball(self):
        """Reset the ball to the center of the screen and reverse its direction."""
        self.ball.goto(0, 0)
        self.ball_dx = abs(self.ball_dx)  # Ensure it moves toward the right wall initially
        self.ball_dy = 3.5
        self.AI_score = 0 # Reset the AI score

    def get_state(self):
        """Get the current state of the game for the AI agent."""
        return [
            self.paddle_left.ycor(),
            self.paddle_left.xcor(),
            self.ball.xcor(),
            self.ball.ycor(),
        ]

    def play_step(self, action):
        """Execute one step in the game with the given action, update the ball position, check for collisions."""
        # Move the paddle based on the action
        if action == 1:
            self.move_paddle_left(up=True)
        elif action == 2:
            self.move_paddle_left(up=False)

        # Update ball position
        self.ball.setx(self.ball.xcor() + self.ball_dx)
        self.ball.sety(self.ball.ycor() + self.ball_dy)

        reward, done = 0, False

        # Border collision for the ball (top/bottom)
        if self.ball.ycor() > 290:
            self.ball.sety(290)
            self.ball_dy *= -1

        if self.ball.ycor() < -290:
            self.ball.sety(-290)
            self.ball_dy *= -1

        # Ball out of bounds - left side (missed by AI paddle)
        if self.ball.xcor() < -390:
            self.reset_ball()
            reward = -25  # Negative reward for missing the ball
            done = True

        # Right side - wall bounce
        elif self.ball.xcor() > 390:
            self.ball.setx(390)
            self.ball_dx *= -1

        # Paddle collision
        if (-340 < self.ball.xcor() < -330) and (self.paddle_left.ycor() - 50 < self.ball.ycor() < self.paddle_left.ycor() + 50):
            self.ball.setx(-330)
            self.ball_dx *= -1
            reward = 5  # Higher reward for successfully hitting the ball
            self.AI_score += 5

        # Update the score display and screen
        self.update_score()
        self.win.update()  # Ensure the screen refreshes with every step

        next_state = self.get_state()
        return reward, done, next_state

    def move_paddle_left(self, up=True):
        """Move the left paddle up or down based on the action, and ensure it stays within bounds."""
        y = self.paddle_left.ycor()
        if up:
            y += 20
        else:
            y -= 20

        # Prevent the paddle from moving out of bounds
        if y > 250:
            y = 250
        elif y < -250:
            y = -250

        self.paddle_left.sety(y)