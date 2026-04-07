import React ,{useState}from "react";
function ModelSelector({models = [], selectedModels = [], setSelectedModels}){

    const [allSelected,setAllSelected]=useState(false);
    const handleSelectAll=()=>{
        if(!allSelected){
           setSelectedModels(models);
        }else{
            setSelectedModels([]);
        }
        setAllSelected(!allSelected);
    };

    const handleChange=(model)=>{
       if(selectedModels.includes(model)){
           setSelectedModels(selectedModels.filter((m)=>m!==model));
       }else{
           setSelectedModels([...selectedModels,model]);
       }
    }


    return (
        <div><label>
            <input type="checkbox" checked={allSelected} onChange={handleSelectAll}/>
        </label>

        <div style={{ display:"grid",
            gridTemplateColumns:"repeat(4,1fr)",
            gap:"8px",
            marginBottom:"16px"}}>

                Select Model:

                    {models.map((model)=>(
                       <label key={model}>
                           <input type="checkbox"
                                  value={model}
                                  checked={selectedModels.includes(model)}
                                  onChange={()=>handleChange(model)}
                              ></input>
                           {model}
                       </label>
                    ))}
        </div>
        </div>
    );
}
export default  ModelSelector;