---
layout: post
title:  "Breaking Down Deepmind's DRAW!"
date:   2016-09-01 11:11:11 +0100
---

In my earliar article we had talked about the Variational autoencoder and had a deep look inside the mathematics that was responsible for VAE's function that was to encode an image and decode from its latent form. Today we will be doing a break down of Deepmind's Deep Recurrent Attentive Writer DRAW . If you haven't gone through my article on VAE, i recommend you to go through it once which would make understanding this paper way to easiar.
Generative models in the recent years.
Test a display math:
$$
   |\psi_1\rangle = a|0\rangle + b|1\rangle
$$
Is it O.K.?
Draw model portrays the essence of how humans have been using attention mechanism to draw a painting . Essentially an artist looks at a part of something that he would like to paint(Say mona lisa) on a canvas and draws the part of it on a part of canvas.There are two things that are happening at the same time:
1.The artist is able to focus only on a part of it which enhances the quality of the drawing since his mental and physical resources are being used to improve the appearance of only a part and not the whole.
2.This allows the artist to be able to improve rest of the drawing by adjusting the rest of strokes required to draw the other parts of the image.
While this technique which is the only way we humans can draw (duh!! since we can’t paint an entire image in one shot) may seem usual with respect to humans but is entirely a different scene when it comes for machines for them to use the same techniques.So the whole point of AI is to bridge the gap between the things that may be difficult for us humans (say calculating 3344543223*32132113123) but would be easy for a machine while a thing such as vision or speech which is easy for humans but is difficult for machines.With respect to this if a machine is able to use attention mechanism to draw things on a canvas it is pretty cool.

The DRAW model essentially has done this quite gracefully.

Lets dive into how exactly the most important component of art generation is devised through the neural networks.

Lets see a few more ways in which DRAW is different from VAE :

1) When we talked about VAE in the last post , we were talking about training of network in n epochs and image generation was just a one step process. But DRAW also involves training of network in n epochs whereas image generation is t time step process. This way DRAW essentially achieves the iterative construction of complex images.This way sketches can be succesfully refined and it allows the DRAW model to suceesively add decoder's output to the distibution that ultimately generates the final image.

2) DRAW uses Recurrent Neural Networks as encoder and decoder, where as VAE used vanilla Feed Forward Neural Networks.

3) An attention mechanism that is allowing the encoder and decoder to focus on certain parts of image to achieve "where to look","where to write" and "what to write".

Lets get back to our favourite oven example(Again this would be a lot easiar if you go through the oven example described in my aricle on VAE) :

Recall that in our oven example in my article on VAE , we had an Oven O1(decoder),Oven O2(encoder),each oven had knobs to tune its settings (theta1,theta2) and Box of inderigents (vector of latent variables)
Imagine a scenario where you ordered some dish and you found it so delicious that after having one spoon of it you realize that you want to recreate the same.How do you do that.

Say you have an Oven O2 which takes your dish that you ordered and gives you out a few inderigents (that were used in making that dish) at every 2 minutes and you are using those 
inderigents simultaneously to cook the same dish back using O1.This is what exactly is happening in our DRAW Model where you have an image(instead of that delicious dish you ordered and wanted to recreate) that you keep encoding using an RNN encoder  (O2) at every t time step and use those latent variables(inderigents) to reconstruct it back to the image using a RNN decoder (O1).

But in a sequential process like this how do you gauge how well you are doing ? One way of doing that is every two minutes you taste both the dish you ordered as well as the dish you are preparing . Along with this you would also need a record of inderigents you already added to your dish you are preparing. Sounds Perfect!!!

As far as gauging this model is considered , at every time step we feed the following information to the RNN encoder (O2) :

1.The hidden state of the encoder (inderigents)
2.The hidden state of the decoder (dish you are preparing) 
3.The Input image (dish you ordered)

Thats it there you have the complete architecture for DRAW Network . Lets look into how it functions :

As we knew our encoder in VAE was responsible for finding out the distribution Q(z|X) , here we find out the distribution of Q(Zt|ht enc) where Zt is the latent vector at time t with respect to given hidden state ht at time t. 

We are chose this ditribution to be gaussian N ( Zt | µt, σt) at time step t.

So at every time step t we are sampling from Zt from Q(Zt | ht enc) and pass it to decoder. Remember how we discussed (in article breaking down VAE)that choosing inderigents Q(Zt) obtained from reverse oven O2 is much easiar than finding inderigents from Bag of inderigents i.e our prior P(Zt). 

The output of the decoder ht dec is added into the canvas matrix Ct via a write operation cumulatively at every time step till T. The final Canvas matrix CT is used to draw the final image.

{% highlight ruby %}
xˆt = x − σ(ct−1)
rt = read(xt, xˆt, ht−1 dec)
h t enc = RNN enc(ht-1 enc, [rt, ht−1 dec])
zt ~ Q(Zt|ht enc)
ht dec = RNN dec(ht-1 dec,zt)
ct = ct−1 + write(ht dec) 
{% endhighlight %}

The paper has beautifully put together the whole process in merely 6 equations.
Lets break down these :

Equation 1 is comparing our original image with the image we are reconstructing at time step t.Here σ is the sigmoid function which converts our canvas matrix into an image reconstructed at time t-1.

In Equation 2 , We are utilizing the knowledge of how well we have reconstructed the image at time t and what we had decoded at the previous time step t-1 via a read function to determine what needs to be read at current time step t. 

This information is passed on to the encoder in Equation 3 , along with what was encoded at previous time step t-1. As our encoder needs to know the status of reconstruction of image along what was encoded and decoded at previous time steps.
With respect to our oven example , we need to tell the reverse oven O2 about how well we are preparing the dish at time step t and what part of the dish we have already recovered, along with what inderigents we have already obtained from O2 at previous time steps. 

This information is again used to calculate the distribution Q(Zt|ht enc) and Equation 4 invloves sampling of zt from distribution Q(Zt|ht enc). This is similar in analogy that during our traning phase , selecting inderigents obtained from reverse oven O2 makes training much simpler than selecting inderigents directly from Bag of inderigents that is our prior P(Zt) which is distributed in a much larger space.

In equation 5 we use decoder RNN to determine the information that needs to written on canvas matrix.we give our oven O1 the inderigents we selected along with the information about the which part of dish it has already recovered at step t-1.This gives the part of dish recovered at time step t. 

In equation 6 we use the write function to write the information given by equation 5 on a canvas matrix Ct. Imagine this process as serving the dish on to the final plate. 

Lets talk a bit about our Loss function : 

The loss function is very similar to VAE's Loss function except here we define our terms at time t.

Total Loss = latent loss + reconstruction loss = − log D(x|cT) + KLD Q(Zt|ht enc)||P(Zt))

here D(x|cT) is a Bernoulli distribution.

Once we have trained the encoder and decoder to do their respective tasks really well, we can move on to the image generation phase where we remove the encoder.
{% highlight ruby %}
Zt ~ P(Zt)
ht dec = RNN dec(ht−1 dec, Zt)
ct = ct−1 + write(ht dec) 
x = D(X|cT ) 
{% endhighlight %}
Here we sample directly from prior P(Zt) and use that to decode and finally write it to canvas matrix iteratively.

Now lets define how are we reading and writing using attention.





{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: http://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
