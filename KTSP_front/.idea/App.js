import React,{useState} from "react";
import FileUpload from "./components/FileUpload";
import ModelSelector from "./components/ModelSelector";
import ConfusionMatrix from "./components/ConfusionMatrix";


function App(){

    const[file,setFile]=useState(null);
    const[model,setModel]=useState("Linear SVM");
    const[results,setResults]=useState(null);

    const runPipeline=async()=>{
        if(!file) return;

        const formData= new FormData();
        formData.append("file",file);
        formData.append("model",model);

        const res=await fetch("http://127.0.0.1:8000/run",{
            method:"POST",
            body:formData,
            });
        const data=await  res.json();
        setResults(data);
    };
    return (
        <div>
            <h1>KTSP pipeline</h1>
            <FileUpload> setFile={setFile}</FileUpload>
            <ModelSelector>setModel={setModel}</ModelSelector>
            <button onClick={runPipeline}>Run Pipeline</button>
            {results &&(
                <>
                <ConfusionMatrix data={results.confusion_matrix}></ConfusionMatrix>
                    <ROCChart data={results.roc_data}></ROCChart>
                </>
            )}
        </div>
    );

}
export default  App;