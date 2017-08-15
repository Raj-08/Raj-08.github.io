---
layout: post
title:  "Deep inside Deep Dream+Code in Tensorflow"
date:   2016-09-01 11:11:11 +0100
---

Visualization or Imagination is a very high form of intelligence possesed by us human biengs. To be able to see things that don't exist requires a very high level of conciousness. These highly intelligent activities can be found in our daily activities:

1.When we want to paint a scenery that we imagine
2.When we interpret cloud shapes as different objects
3.When we dream

All of these activities require our brain to do something different rather than just interpret the visual information it is recieving. When we see a cloud for example we may say that it looks like a Dog or like an Elephant. Clearly they aren't actual Dogs or Elephants flying in the skies but they have some similarity. Even though this similarity is minimal,our Brain somehow manage to connect the missing dots to form a complete picture.

So when we see a cloud and we interpret it to something , our brain essentially kept adding information to the visual scene(imagination) that we can't see. So we need to see what is happening in our brain exactly. While this is an abstruse concept we can use Convolutional Neural Networks to know more about what is happening between layers of our Brain. 

The layers in CNN store informations related to the things it is trained to recognize.Infact we know that the starting layers of CNN store very minimal details like strokes , lines, shapes , edges and contours. And the final layers of the CNN use those details provided by the initial layers to form a more defined image. Hence these layers essentially stores the visual information.

So lets give a trained CNN an image. It is pretty much analogous to the case when we are looking at the clouds. Then we pick a layer from the CNN and enhance what it has detected so far. By enhancing we mean that we want to see what its activations are for an entire layer , by backproping the gradient of that activation back to the input image.Gradient is a measure how things are changing or in our case we mean that gradient is a way of visualization or imagination.

So when we were doing the task of recognition using a CNN which is a discriminative process and we essentially wanted what category does a particular image belong to. It involved gradient descent and we wanted to minimize the cost function .But here we perform gradient ascent and maximize the layer's activation. So essentially we will be following the direction of the gradient(ascent) and will be maximizing the selected layer or neuron's activation by changing our input image that is by adding the gradient back to the image.

Lets impliment it in tensorflow and see the results for ourselves.
Lets import the inception model :

{% highlight ruby %}
from libs import inception
net = inception.get_inception_model()
{% endhighlight %}

Now we will get the inception graph:

{% highlight ruby %}
tf.import_graph_def(net['graph_def'], name='inception')
g = tf.get_default_graph()
names = [op.name for op in g.get_operations()]
print(names)
{% endhighlight %}










As we know a Convolutional Neural Network is made of up different layer:
1.Convolution layer
2.Max Pool Layer
3.Dropo

These layers store informations related to the things it is trained to recognize.Infact we know that the starting layers of CNN store very minimal details like strokes , lines, shapes , edges and contours. And the final layers of the CNN use those details provided by the initial layers to form a more defined image. Hence these layers essentially stores the visual information.


How do we come to the conclusion is very abstract and what is happening 
As we know 




Here we are using a pretrained model by Google called the Inception V5 . This is implimented using Tensorflow.

Lets dig in :)

Lets import the inception model :

{% highlight html %}
from libs import inception
net = inception.get_inception_model()
{% endhighlight %}

Now we will get the inception graph:

{% highlight html %}

device = '/cpu:0'

g = tf.Graph()

with tf.Session(graph=g) as sess, g.device(device):
    
    # Now load the graph_def, which defines operations and their values into `g`
    tf.import_graph_def(net['graph_def'], name='net')
{% endhighlight %}

Let's see all the operations that exist in this graph:

{% highlight html %}
names = [op.name for op in g.get_operations()]
print(names[0])
{% endhighlight %}

Now we need to select the layers ,

{% highlight html %}
names = [op.name for op in g.get_operations()]
print(names[0])
{% endhighlight %}

And select the first layer to which the gradients will be added:

{% highlight html %}
x = g.get_tensor_by_name('net/images:0');
{% endhighlight %}

As we know we want to calculate the gradient of layers with respect to the input layer and add the gradients back to the input image. For this purpose we use Tensorflow's tf.gradient to caluclate the differential between the two. This is effectively telling us which pixels contribute to the predicted layer, class, or given neuron with the layer.Hence we define a plot_gradient function that takes the input image , input layer and the layer we selected.

{% highlight html %}
def plot_gradient(img, x, feature, g, device='/cpu:0'):
    with tf.Session(graph=g) as sess, g.device(device):
        saliency = tf.gradients(tf.reduce_mean(feature), x)
        this_res = sess.run(saliency[0], feed_dict={x: img})
        grad = this_res[0] / np.max(np.abs(this_res))
        return grad
{% endhighlight %}

Now its time to define the Dream function that will do all the dreamy work. Our Gradients represent the direction we should move our input in order to meet our objective stored in "gradient"
    

{% highlight html %}
def dream(img, gradient, step, net, x, n_iterations=50, plot_step=10):
    img_copy = img.copy()

    fig, axs = plt.subplots(1, n_iterations // plot_step, figsize=(20, 10))

    with tf.Session(graph=g) as sess, g.device(device):
        for it_i in range(n_iterations):
            this_res = sess.run(gradient[0], feed_dict={x: img_copy})[0]

            # Let's normalize it by the maximum activation
            this_res /= (np.max(np.abs(this_res) + 1e-8))
            
            # Adding the gradient back to the input image
            img_copy += this_res * step

            # Plot the image
            if (it_i + 1) % plot_step == 0:
                m = net['deprocess'](img_copy[0])
                axs[it_i // plot_step].imshow(m)
{% endhighlight %}







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
