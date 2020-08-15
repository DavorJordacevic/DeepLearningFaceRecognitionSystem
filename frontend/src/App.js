import React, {Component} from 'react';
import './App.css';
import axios from 'axios';

class App extends Component {

  state = {
    selectedFile : null
  }

  fileSelectedHandler = event => {
    //console.log(event.target.files[0]);
    this.setState({
      selectedFile : event.target.files[0]
    })
  }

  fileUploadHandler = () => {
      // HTTP request using axios
      const formData = new FormData();
      formData.append('image', this.state.selectedFile, this.state.selectedFile.name);
      axios.post('http://localhost:5000/identification', formData)
        .then(res => {
          console.log(res);
        })
  }

  render() {
    return (
      <div className="App">
       <input type="file" onChange={this.fileSelectedHandler} />
       <button  onClick={this.fileUploadHandler}>Upload</button>
      </div>
    );
  }
}
export default App;
