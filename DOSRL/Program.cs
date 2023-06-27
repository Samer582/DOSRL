using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RLDoS
{
    class Program
    {
        // Policy iterations
        const int POLICY_ITERATIONS = 100;

        // Learning rate
        static readonly double LEARNING_RATE = 0.1;

        // Discount factor
        static readonly double DISCOUNT_FACTOR = 0.9;

        // Reward for taking correct action
        static readonly double POSITIVE_REWARD = 1;

        // Reward for taking incorrect action
        static readonly double NEGATIVE_REWARD = -1;

        // States
        enum State { ATTACK, NORMAL };

        // Actions
        enum Action { BLOCK, ALLOW };

        static readonly Random random = new Random();
        class QLearningAgent
        {
            // Q-Value matrix
            Dictionary<State, Dictionary<Action, double>> qValues;

            // Initialize Q-Value matrix
            public QLearningAgent()
            {
                qValues = new Dictionary<State, Dictionary<Action, double>>();
                qValues.Add(State.ATTACK, new Dictionary<Action, double>());
                qValues[State.ATTACK].Add(Action.BLOCK, 0.0);
                qValues[State.ATTACK].Add(Action.ALLOW, 0.0);

                qValues.Add(State.NORMAL, new Dictionary<Action, double>());
                qValues[State.NORMAL].Add(Action.BLOCK, 0.0);
                qValues[State.NORMAL].Add(Action.ALLOW, 0.0);
            }

            // Get Q-Value for a state and action
            public double GetQValue(State state, Action action)
            {
                return qValues[state][action];
            }

            // Set Q-Value for a state and action
            public void SetQValue(State state, Action action, double value)
            {
                qValues[state][action] = value;
            }

            // Choose an action based on epsilon-greedy policy
            public Action ChooseAction(State state, double epsilon)
            {
                Action choice;
                double randomValue = random.NextDouble();
                if (randomValue < epsilon)
                {
                    // Choose random action
                    int actionIndex =Convert.ToInt16(random.NextDouble()) ;
                    choice = (Action)actionIndex;
                }
                else
                {
                    // Choose action with highest q-value
                    double maxQ = double.MinValue;
                    Action maxA = Action.BLOCK;

                    foreach (Action action in qValues[state].Keys)
                    {
                        double qVal = GetQValue(state, action);
                        if (qVal > maxQ)
                        {
                            maxQ = qVal;
                            maxA = action;
                        }
                    }
                    choice = maxA;
                }
                return choice;
            }

            // Train the agent
            public void Train(State state, Action action, double reward, State newState)
            {
                // Calculate temporal difference
                double currentQ = GetQValue(state, action);
                double newQ = reward + DISCOUNT_FACTOR * GetQValue(newState, ChooseAction(newState, 0));
                double tdError = newQ - currentQ;

                // Update Q-Value 
                SetQValue(state, action, currentQ + LEARNING_RATE * tdError);
            }
        }

        static void Main(string[] args)
        {
            // Create Q-Learning agent
            QLearningAgent agent = new QLearningAgent();

            // Training loop
            for (int i = 0; i < POLICY_ITERATIONS; i++)
            {
                // Choose a random state
                State state = (State)random.NextDouble();

                // Choose an action based on epsilon-greedy policy
                Action action = agent.ChooseAction(state, 0.1);

                // Get reward and new state
                double reward; State newState;
                if (state == State.ATTACK && action == Action.BLOCK)
                {
                    reward = POSITIVE_REWARD;
                    newState = State.NORMAL;
                }
                else if (state == State.ATTACK && action == Action.ALLOW)
                {
                    reward = NEGATIVE_REWARD;
                    newState = State.ATTACK;
                }
                else
                {
                    reward = 0;
                    newState = state;
                }

                // Train the agent
                agent.Train(state, action, reward, newState);
            }

            Console.WriteLine("Training complete!");
            Console.ReadKey();
        }
    }
}