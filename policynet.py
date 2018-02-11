import tensorflow as tf
import numpy as np
import feedforward

class policynet:

    def __init__(self, hidden_units, num_actions):

       # self.actions_i = []
       # self.rewards_i = []
       # self.observations_i = []
        self.baseline = 10

        self.observations = tf.placeholder(tf.float32, shape=(None, 4), name="observations")
        self.actions = tf.placeholder(tf.int32, shape=None, name="actions")  # integer actions
        self.returns = tf.placeholder(tf.float32, shape=None, name="returns")

        self.rewards = tf.placeholder(tf.float32, shape=None, name="rewards")
        episode_reward = tf.reduce_sum(self.rewards)

        hidden = tf.layers.dense(inputs=self.observations, units=hidden_units, activation=tf.nn.relu)
        logits = tf.layers.dense( inputs=hidden, units=num_actions )

        self.policy = tf.nn.softmax(logits)

        one_hot_actions = tf.one_hot(indices=tf.cast(self.actions, tf.int32), depth=2)

        self.loss = tf.reduce_sum(-tf.log( tf.reduce_sum(one_hot_actions * self.policy, 1 ) ) * (self.returns - self.baseline))

        #self.loss = episode_reward * tf.losses.softmax_cross_entropy(one_hot_actions, logits, weights=(self.returns - self.baseline))

        #episode_reward * tf.reduce_sum(tf.log(self.policy * one_hot_actions))


        #Thisis probably wrong.  Need to understand algorithm properly first then fix.
        self.train_op = tf.train.AdamOptimizer(0.01).minimize(self.loss)
      #  self.train_op = optimizer.minimize(
      #      loss=self.loss,
      #      global_step=tf.train.get_global_step())

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def update_policy(self, observations, actions, rewards):

        self.baseline = 10#0.9*self.baseline + 0.1 * sum(rewards)


        #calculate returns
        returns = np.matmul( np.triu(np.ones([len(rewards),len(rewards)])) , rewards)

       # print(returns)

        #This is probably
        [_, l] = self.sess.run([self.train_op, self.loss], feed_dict={
                                                   self.observations: observations,
                                                   self.actions: actions,
                                                   self.returns: returns,
                                                   self.rewards: rewards
                                                   })

        return l

    def best_action(self, observation):

        pi = self.sess.run(self.policy, feed_dict={
            self.observations: [observation]
        })

        return np.argmax(pi)

    def sample_action(self, observation):
        pi = self.sess.run(self.policy, feed_dict={
            self.observations: [observation]
        })

        return np.random.choice([0,1], 1, p=pi[0])[0]
