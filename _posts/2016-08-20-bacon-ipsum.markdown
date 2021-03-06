---
layout: post
title:  "Mathematics of Variational Autoencoders made really easy!"
date:   2016-09-01 11:11:11 +0100
---
This article is intended for those who lack a statistical background and and are unfamiliar with terms used in probability.Although you can find great explainations to Varitaional Autoencoders and its implimentations (links can be found at the bottom), its easy to get confused with the term used. Here i have tried to make it as simple as possible to understand the main mathematical concepts behind VAE and provide with the necessary intution that is needed. This will also help in understanding the basics of generative modelling which is required for breaking down of more interesting papers which i will be doing in my upcoming articles.

Lets us take the following image. An image is simply millions of pixels put together. When you see this image , there is a particular arrangements of the pixels that is making it look as it is. That means there is dependecy of of pixels in nearby space.So given an image that looks convincingly real have a high P(X) while something that is noise should have a low value.
There is high P(X) for that one combination of pixels that makes sense when compared to other combinations that doesn't.

When we talk about the generative modelling of MNIST dataset it is all about teaching the neural network the art of drawing numbers [0-9]. That is it needs to have a sense of how to draw a digit. Say when it needs to generate a 2 , it needs to have an intution that forces it to draw the "2" in that particular manner. It needs to draw a small curve at the top and then extend it to the bottom left in a curved manner and finally draw a line to the right. This information or intution is what is contained in a space called Latent variables. Think of latent space as a space of sparse values that are responsible for producing different kind of digits. One setting of that latent space may be responsible for generating a "1" , other setting may be responsible for generating a "2". When we say we a well defined latent space we mean that every data in our dataset can be generated by some setting of that latent space.

Let me make things clear by introducing a very simple case. Imagine you have three things:

A Cooking Oven O1 | Oven's Settings | Box of Inderigents

When you put a couple of inderigents in the oven O1 you get a type of dish on the other side.
Say you are a lazy person and you don't want to play with the knobs everytime you want a new dish so basically what you want is an ideal setting / tuning that works for a multitiude of dishes.So that every time you choose different inderigents from the Box of inderigents you get a different dish but with the same settings of the oven.(Interesting right!)

This is all that is happening in our case except that the oven is a deterministic function f    (which is the neural network), oven's settings are parameters [theta] and box of inderigents is our vector of latent variables.
The dish in our case are the reconstructed digits [0-9].

Lets introduce some probability to this case.

When you pick some inderigents from the Box of inderigents you need to have a high probability that cooking of those inderigents would lead to a specific dish.This probability is a Prior. A prior is the information you have before you make an inference about an event, or in our case this probability is letting us know how confident are we that these inderigents would lead to the dish we want.

A Bayesian would typically say :

P(X) = integral (P(X|z; θ)P(z))dz.

P(X) is the probability of the dish. [posterior]
P(z) is the probability of our inderigents. [prior]
P(X | z;θ) is the deterministic function f.[function approximateed by our neural network]

P(X | z;θ) defines the dependence of z that leads to generation of a specific X. This is also defined by a probability distribution which is Gaussian.

P(X | z; θ) = N (X | f(z; θ), σ^2 * I). 

This distribution has mean f(z; θ) and covariance equal to the identity matrix I times some scalar σ. 

Early in our training our model would fail to produce good quality X's but as we apply gradient descent on the gaussian distribution we increase the likelihood of some data given a latent vector.

Lets get back to our oven example.

Lets say you have another oven O2 but its function is reversed.In this oven you put a dish and it gives you out the inderigents.This is a great way of finding out what exact inderigents would lead to a perfectly good dish.  

This O2 is our case is the encoder which is converting our perfectly good digits into a latent vector and O1 is a our decoder which converts those latent vector into reconstructed digits. Both of these encoder and decoder are neural networks. Where the job of the encoder is to map input X into z via a probability distribution function Q(z|X) and the job of decoder is to map latent vector z into reconstructed X via probability distribution P(X|z).

What we need to note here is given a set of random variables that are distributed normally , we can map it into a much complicated structure via some function approximater. As we know neural networks are really good at approximating complicated functions we use them to map our indipendent and identically distributed z's to whatever latent vector that is needed and then later use those to map into complex distribution of pixels that define an image.

With our decoder those normally distributed z's gets converted into latent information about the stroke , angle of the digit(image) and later layers can use those latent values to render a fully reconstructed image.

Back to our oven example:

Say among 10 dishes that you have learnt you want to make a particular one. If i tell you to open your Bag of inderigents and select a few to make a particular dish , it will be complicated for you to select the exact set of inderigents among the rest in the Bag of inderigents.However if i tell you that you can use oven O2 where you pass the dish you want to make, determine the inderigents and select inderigents from there, it would be much easier for you in your training phase and you would be able to make a dish that is pretty close to what you desired by using those in oven O1. By "training phase" i mean that while we are training, both are encoder and decoder are weak at their tasks in the begninning.So in this stage , providing samples to decoder from distribution Q(z|X) ,which is calculated by the encoder,  would be much efficient rather than providing it from prior P(z) which is much larger space.
However once our network becomes fully trained, we can strip off the encoder and use the trained decoder to directly generate images ("generation phase")from latent vectors sampled from prior P(z) and thus fulfilling the purpose of VAE being a generative method. 

The process of selection of inderigents from the Bag is called Sampling. And this example above justifies our need for encoder. We determine Q(z|X) using encoder and sample z from this distribution rather than directly sampling from P(z).

Now that we have understood the process lets look into how the Bayesians are making all this happen on the inside.

Lets understand what KL Divergence exactly means ?

We have distribution P which is our true distribution and distribution Q which is our approximating distribution.Remember as we described above the need of Q , so we need this distribution to be as close as possible to be to that of distribution of P.Think of it as a way of O2 trying the best possible way of picking inderigents from a hypothetical way of perfect picking inderigents required for a dish . The way we do is , is by defining KL Divergence. KL Diverence lets you calculate exactly the information lost when you are trying to approximate one distribution (Q(z|X)) over a true distribution (P(z)).And our cost function will make sure this divergence is minimized. KL Divergence measures this value in nats.

KL Divergence is defined as the following :

KL(p||q)=∑p(x)log p(x)/q(x)

With respect to our Oven Example as i had told you O2 is letting you discover the inderigents behind the dish Q(z|X). But assuming it doesn't know how to do it in the beginning we try to teach it by tuning its knobs to get it to learn from our manual way of choosing inderigents from the Bag that is our prior P(z).

Lets start by defining relationship between Q(z) and P(z|X) using the Kullback Liebler Divergence.

KLD [Q(z)|| P(z|X)] = E z~Q [log Q(z) − log P(z|X)] 

Applying Bayes rule to the above 

KLD [Q(z)|| P(z|X)] = E z~Q [log Q(z) − log P(X|z)- log P(z)] + log p(X)

After multiplying the above equation by -ve sign we get

-KLD[Q(Z) || P(Z|X)] = E z~Q[- Log Q(Z) + Log P(X|Z) + Log P(Z)] - log P(X)

Since Log P(X) is indipendent of z we transfer it to LHS 

Log P(X) - KLD[Q(Z) || P(Z|X)] = E z~Q[- Log Q(z) + Log P(X|z) + Log P(z)]

Log P(X) - KLD[Q(Z) || P(Z|X)] = E z~Q[ Log P(X|Z)] - KLD[Q(z) || P(z)]

{How we reduced to the above equation is via KL(Q(z)||P(z))=∑ Q(z)log Q(z)/P(z) 
As there was an expectation of Q , we could omit the Q(z) to reduce to the form 
KL(Q(z)||P(z))}

As we had discussed above @ *, we conclude that Q(Z) is much smaller if we condition it with X

Log P(X) - KLD[ Q(Z) || P(Z | X) ] = E z~Q [ Log P(X|Z) ] - KLD[Q(Z|X) || P(Z)]

Right hand side looks a lot like variational autoencoder because now Q is encoding X into Z and P is decoding it to reconstruct X.

While this arrangement would work perfectly fine during the forward propogation , this would fail badly during the back ward propogation.Its because to be able to train a neural network, all your layers should be differentiable and thereby letting the gradients/error flow from one end to the other.However in our process we have to sample z from Q(z|X) which is a non continous operation and has no gradient.To remove this impediment we use the reparameterization trick. 
So earliar we were sampling from `Q(z|x)~N(μ,σ^I)` but now we sample `ϵ~N(0,I)` and calculate 
{% highlight ruby %}
z = μ + σ * ϵ
{% endhighlight %}
This is almost as close to magic and this makes the whole network differentiable.
This can be expressed as sum of latent loss and reconstruction loss. 
where reconstruction loss is -Log P(X|Z) and latent loss is KLD[Q(Z|X) || P(Z)].
Latent loss is given by 1/2∑(1+log(σ^2)−μ^2−σ^2)
so our total cost function becomes 

COST = -log P(X|Z) + 1/2∑(1+log(σ^2)−μ^2−σ^2)

Here we are learning both the abilities of encoder to encode as well as decoder to decode.
However variational autoencoder is a generative process and its ultimate goal is to generate images from latent space.

The whole point of using a reduced space of distribution Q(z|X) ,O2, was to ease the training process where in we were escaping the tedious task of sampling directly from prior P(Z) when our decoder wasn't even good at recovering images from latent vectors.
But once the training is completed we no longer the need the encoder , and instead can sample directly from prior P(z). With respect to our oven example we can remove O2 and simply use O1, where we select the inderigents from the Bag of inderigents directly and use O1 which is fully trained now,to prepare the dish.  


To add new posts, simply add a file in the `_posts` directory that follows the convention `YYYY-MM-DD-name-of-post.ext` and includes the necessary front matter. Take a look at the source for this post to get an idea about how it works.

Jekyll also offers powerful support for code snippets:

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
