<h1>Image Classification of Fresh and Rotten Fruits using Deep Learning</h1>

<h2 id="overview">Overview</h2>
<p>This project focuses on classifying fruits as either fresh or rotten using deep learning techniques. The primary objective is to provide a solution that aids consumers in determining the quality of fruits they purchase.</p>

<h2 id="authors">Authors</h2>
<ul>
  <li>Arvin Yuwono</li>
  <li>Christopher Owen</li>
  <li>Justin Theofilus Yonathan</li>
  <li>Pandya Limawan</li>
</ul>

<h2 id="table-of-contents">Table of Contents</h2>
<ul>
  <li><a href="#overview">Overview</a></li>
  <li><a href="#authors">Authors</a></li>
  <li><a href="#table-of-contents">Table of Contents</a></li>
  <li><a href="#problem-description">Problem Description</a></li>
  <li><a href="#solution-features">Solution Features</a></li>
  <li><a href="#solution-design-architecture">Solution Design Architecture</a></li>
  <li><a href="#appendices">Appendices</a></li>
</ul>

<h2 id="problem-description">Problem Description</h2>
<p>Consumers tend to find it challenging to differentiate the freshness of fruits by mere visual inspection. This project aims to address this issue by leveraging deep learning to classify fruits as either fresh or rotten based on given images.</p>

<h2 id="solution-features">Solution Features</h2>
<p>The algorithm implemented in this project is MobileNetV2, an efficient deep learning model optimized for mobile devices. Additional technologies used in this project include machine learning frameworks like Keras.</p>
<p>User Interface is built using the web framework FastAPI and Bootstrap for creating responsive websites. Render is utilized for cloud-based deployment.</p>

<h2 id="solution-design-architecture">Solution Design Architecture</h2>
<p>Datasets are obtained from Kaggle, which are then organized into folders based on their condition. To perform data augmentation, Keras' ImageDataGenerator was applied to enhance the model's performance.</p>
<p>MobileNetV2 was chosen as the base model for transfer learning. The model was compiled with the Adam optimizer, cross-entropy loss function, and accuracy metric. The training process involved training the model on 80% of the dataset for 100 epochs with early stopping enabled.</p>

<h2 id="appendices">Appendices</h2>
<p>Users can access the web interface by visiting <a href="https://fresh-rotten-fruits-classifier.onrender.com/">https://fresh-rotten-fruits-classifier.onrender.com/</a>. For more detailed instructions, visit the report <a href="report.pdf">here</a>.</p>

<h2>Video Demo</h2>
<p>A demonstration of the project can be viewed <a href="#https://youtu.be/ZEOwV2rRGKM">here (YouTube link)</a>.</p>
