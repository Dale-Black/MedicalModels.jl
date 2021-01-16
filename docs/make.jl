using MedicalModels
using Documenter

makedocs(;
    modules=[MedicalModels],
    authors="Dale <djblack@uci.edu> and contributors",
    repo="https://github.com/Dale-Black/MedicalModels.jl/blob/{commit}{path}#L{line}",
    sitename="MedicalModels.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Dale-Black.github.io/MedicalModels.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Dale-Black/MedicalModels.jl",
)
