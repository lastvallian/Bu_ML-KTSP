import React,{useState} from 'react';
import axios from'axios';
const FileUpload=()=>{
    const[selectFile,setSelectFile]=useState(null);
    const[isUploading,setIsUploading]=useState(false);

    //Function to handle file selection
    const handleFileChange=(event)=>{
        setSelectFile(event.target.files[0]);
    };
    //Function to handle the upload to the server
    const handleFileUpload=async()=>{
        if(!selectFile){
            alert("Please select a file first!");
            return;
        }
        setIsUploading(true);
        const formData=new FormData();
        formData.append('file',selectFile);

        try{
            const  response =await axios.post('API',formData,{
                headers:{
                    'Content-Type':'multipart/form-data',
                },
            });
            const data=await  response.json();
            onUploadSuccess && onUploadSuccess(data);
        }catch(err){
            setErro(err.message);
        }finally{
            setLoading(false);
        }


    };
    return(
        <div style={{borser:"1px solid#ccc",padding:"16px"}}>
            <h3>Upload File</h3>
            <input type="file" onChange={handleFileChange}></input>
            {file && <p>Selected :{file.name}</p>}
            <button onClick={handleFileUpload} disabled={loading}>
                {loading?"Uploading...":"Upload"}
            </button>
            {error && <p style={{color:"red"}}>{error}</p>}
        </div>
    );
}
export  default  FileUpload;