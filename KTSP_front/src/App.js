import React,{useState,useEffect} from "react";
import FileUpload from "./components/FileUpload";
import ModelSelector from "./components/ModelSelector";
import PCASelector from "./components/PCASelector";
import {ROCChart,ConfusionMatrix} from "./components/Visualization";
function App() {
   const models=["Linear SVM","RBF SVM","Decision Tree","Naive Bayes","kNN","KTSP","Custom SVMTrainer"];

   const[selectedModels,setSelectedModels]=useState([]);
   const[usePCA,setUsePCA]=useState(false);
   const [uploadResult,setUploadResult]=useState(null);
   const [modelResults,setModelResults]=useState(null);
   const[unloading,setUnloading]=useState(null);
   //Adding for in
   const[inferenceModel,setInferenceModel]=useState(null);
   const[predictionResult,setPredictionResult]=useState(null);
   const[testFile,setTestFile]=useState(null);
   const[isInferenceLoading,setIsInferenceLoading]=useState(false);
   const[experimentName,setExperimentName]=useState(false);

    // For state of Windows open or not
    const [isModalOpen, setIsModalOpen] = useState(false);
   // For name of hostname
    const currentHostname = window.location.hostname;
   //Using the API_BASE_URL
   let API_BASE_URL;

   if (currentHostname.includes("localhost") || currentHostname.includes("127.0.0.1")) {
    // Using Localhost if host in localmachein
    API_BASE_URL = "http://localhost:8000";
  } else {
    // IFin GitHub Codespaces cloud ,Automatically replace -3000 for -8000
    const backendHostname = currentHostname.replace("-3000", "-8000");
    API_BASE_URL = `https://${backendHostname}`;
  }

   //sychn for Upadating modelResults when updating uploadResult
    useEffect(()=>{

        if(uploadResult){
            //if return array setting up for array if for {results:{...}} for results
            const resultsArray=Array.isArray(uploadResult)?uploadResult:(uploadResult.results || []);
            setModelResults(resultsArray);
        }
    },[uploadResult]);

    // using：call the function to prediction
    const openPredictionModal = (model) => {
        setPredictionResult(null);
        setTestFile(null);
        setInferenceModel(model);
        setIsModalOpen(true); // set as true
    };

    //Adding prediction function
    const handleRunPrediction =async()=>{
      const folderName = inferenceModel?.model_class || inferenceModel?.model;
      if(!testFile||!folderName){
          alert("Please select a file and a trained model first!");
          return;
      }

      setIsInferenceLoading(true);
      const formData=new FormData();
      formData.append("file",testFile);
      formData.append("folder_name", folderName);
      try{
         const response = await fetch(`${API_BASE_URL}/predict`,
         { method:"POST",
           body: formData,
         });
         const resData=await response.json();

         if (resData.status === "success") {
               //resData.data is results for backend result
               setPredictionResult(resData.data);

             } else {
               alert("prediction failure: " + resData.detail);
             }

      }catch(error){
        console.error("Prediction failed:", error);
        alert("Prediction error, check console.");
      }finally{
        setIsInferenceLoading(false);
      }
    }

   return(
       <div style={{padding:"24px"}}>
         <h2>KTSP model analyzer</h2>
         <ModelSelector models={models} selectedModels={selectedModels}
         setSelectedModels={setSelectedModels}></ModelSelector>
         <PCASelector usePCA={usePCA} setUsePCA={setUsePCA}></PCASelector>
         <div style={{ marginBottom: "20px", padding: "10px", background: "#f9f9f9", borderRadius: "8px" }}>
           <label><strong>1. Define Experiment Name: </strong></label>
           <input type="text" value={experimentName || ""}
           placeholder="e.g., Hospital_Batch_01" onChange={(e)=>setExperimentName(e.target.value)}
           style={{ marginLeft: "10px", padding: "5px", width: "250px" }}></input>
         </div>
         <FileUpload
             selectedModels={selectedModels} models={models}
             usePCA={usePCA}
             onUploadSuccess={setUploadResult}>
             experimentName={experimentName}</FileUpload>
         {
           uploadResult&&(
                 <div style={{marginTop:"20px"}}>

                   <h4> Upload the result </h4>
                     <table style={{ width: "100%", borderCollapse: "collapse", marginTop: "10px" }}>

                         <thead>
                         <tr style={{ backgroundColor: "#f4f4f4", textAlign: "left" }}>
                             <th style={{ padding: "10px", border: "1px solid #ddd" }}>Model</th>
                             <th style={{ padding: "10px", border: "1px solid #ddd" }}>Accuracy</th>
                             <th style={{ padding: "10px", border: "1px solid #ddd" }}>AUC</th>
                         </tr>
                         </thead>
                         <tbody>
                         {modelResults &&modelResults.map((r, index)=>(
                             <tr key={r.model_class || index}>
                                 <td style={{ padding: "10px", border: "1px solid #ddd" }}>{r.model_name}</td>
                                 <td style={{ padding: "10px", border: "1px solid #ddd" }}>{typeof r.performance?.accuracy === 'number'
                                                                                                    ? r.performance.accuracy.toFixed(3)
                                                                                                    : "N/A"}</td>
                                 <td style={{ padding: "10px", border: "1px solid #ddd" }}>
                                    {typeof r.performance?.auc === 'number'
                                             ? r.performance.auc.toFixed(3)
                                             : "N/A" || "N/A"}
                                 </td>
                                 <td style={{ padding: "10px", border: "1px solid #ddd" }}>
                                     {/* call openPredictionModal */}
                                     <button
                                         onClick={() => openPredictionModal(r)}
                                         style={{ cursor: 'pointer', backgroundColor: '#e7f3ff', border: '1px solid #007bff', borderRadius: '4px', padding: '5px 10px' }}
                                     >
                                         Select For Prediction
                                     </button>
                                 </td>
                             </tr>
                         ))}
                        </tbody>
                     </table>
                      /*Interfernce model region */
                     {isModalOpen && inferenceModel&&(
                              <div style={{  marginTop: "30px", padding: "20px", backgroundColor: "#f0f7ff", borderRadius: "8px", border: "1px solid #007bff" }}>
                                  <button
                                      onClick={() => {
                                          setIsModalOpen(false);      // closing dialog
                                          setPredictionResult(null);  // clear the result and reset
                                      }}
                                      style={{ padding: "5px 10px", cursor: "pointer", backgroundColor: "#ff4d4f", color: "#fff", border: "none", borderRadius: "4px" }}
                                  >
                                      close (Close)
                                  </button>
                                  <h3>Real time Prediction Console</h3>

                                <p> <strong>Selected Model:</strong>
                                     {inferenceModel.model}
                                    </p>
                                    <input type="file" onChange={(e)=>setTestFile(e.target.files[0])}/>
                                    <button onClick={handleRunPrediction} disabled={isInferenceLoading}
                                     style={{ marginLeft: "10px", padding: "8px 16px", backgroundColor: "#28a745", color: "#fff", border: "none", borderRadius: "4px", cursor: "pointer" }}>
                                        {isInferenceLoading?"Predicting":"Run Interference"}
                                    </button>
                                    {predictionResult&&(
                                    <div style={{ marginTop: "15px", padding: "10px", background: "#fff", border: "1px solid #ccc" }}>)}
                                    <strong>Result:</strong>{JSON.stringify(predictionResult.Prediction_Result)}<br/>
                                   <div style={{ maxHeight: "300px", overflowY: "auto" }}>
                                   <table style={{ width: "100%", borderCollapse: "collapse" }}>
                                      <thead>
                                      <tr style={{ backgroundColor: "#f8f9fa" }}>
                                       <th style={{ border: "1px solid #dee2e6", padding: "8px" }}>Sample No</th>
                                       <th style={{ border: "1px solid #dee2e6", padding: "8px" }}>Prediction Category</th>
                                         <th style={{ border: "1px solid #dee2e6", padding: "8px" }}>Confidence</th>
                                       </tr>
                                      </thead>
                                      <tbody>
                                       {/* Assuming Prediction_Result is array */}
                                      {Array.isArray(predictionResult.Prediction_Result)?(
                                       predictionResult.Prediction_Result.map((res,idx)=>(
                                          <tr key={idx}>
                                              <td style={{ border: "1px solid #dee2e6", padding: "8px", textAlign: "center" }}>{idx + 1}</td>
                                              <td style={{ border: "1px solid #dee2e6", padding: "8px", fontWeight: "bold", color: "#007bff" }}>
                                                                {res}
                                              </td>
                                             <td style={{ border: "1px solid #dee2e6", padding: "8px" }}>
                                             {predictionResult.Confidence_Score&&predictionResult.Confidence_Score[idx]? (predictionResult.Confidence_Score[idx] * 100).toFixed(2) + "%"
                                                                                                                               : "N/A"}
                                             </td>
                                           </tr>
                                         ))):
                                         /*for Single Data*/
                                         <tr>
                                          <td style={{ border: "1px solid #dee2e6", padding: "8px", textAlign: "center" }}>1</td>
                                          <td style={{ border: "1px solid #dee2e6", padding: "8px", fontWeight: "bold" }}>
                                             {predictionResult.predictionResult||"UNKNOWN"}
                                          </td>
                                          <td>{predictionResult.Confidence_Score ? (predictionResult.Confidence_Score * 100).toFixed(2) + "%" : "N/A"}</td>
                                         </tr>

                                         }
                                      </tbody>
                                   </table>
                                   </div>
                                </div>

                       )}
                       </div>
                        )}

                     <details style={{ marginTop: "20px" }}>
                         <summary>View Raw JSON Data</summary>
                         <pre style={{ backgroundColor: "#eee", padding: "10px" }}>
                             {JSON.stringify(uploadResult, null, 2)}
                         </pre>
                     </details>
                     <pre>{JSON.stringify(uploadResult,null,2)}</pre>

                     <div style={{ marginTop: "40px" }}>
                         {modelResults &&modelResults.map((r, idx) => (
                             <div key={r.model || idx} style={{
                                 marginBottom: "50px",
                                 padding: "20px",
                                 border: "1px solid #eee",
                                 borderRadius: "8px"
                             }}>
                                 <h3 style={{ borderBottom: "2px solid #333", paddingBottom: "5px" }}>
                                     Model Details: {r.model}
                                 </h3>
                             <div style={{display: "flex", flexWrap: "wrap", gap: "40px", marginTop: "20px"}}>
                                 <div style={{ flex: "1", minWidth: "300px" }}>
                                       <h4>Roc Curve</h4>
                                     {(r.visualization?.roc || r.roc)?(<ROCChart data={r.visualization?.roc || r.roc}></ROCChart>):(<p style={{ color: "#888" }}>No ROC data available</p>)}
                                 </div>
                                 /**Confusion_Martix>
                                 <div style={{ flex: "1", minWidth: "300px" }}>
                                     <h4>Confusion Matrix</h4>
                                     {(r.confusion_matrix||r.visualization?.confusion_matrix)?(
                                         <ConfusionMatrix matrix={r.confusion_matrix|| r.visualization?.confusion_matrix}></ConfusionMatrix>
                                     ) : (
                                         <p style={{ color: "#888" }}>No Matrix data available</p>
                                     )}
                                 </div>
                                /** HeatMap Visualization**/
                         </div>
                         {r.visualization?.heatmap&&(
                             <div style={{ marginTop: "30px" }}>
                                 <h4>Feature Heatmap</h4>
                                 <p>Heatmap data received: {r.visualization.heatmap.columns.length} features</p>
                             </div>
                         )}
                             </div>
                         ))}
                     </div>
                 </div>
             )}
       </div>
   );
}

export default App;
