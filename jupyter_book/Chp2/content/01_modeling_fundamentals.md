# Modeling Fundamentals

## Introduction

Data-driven methodologies are currently revolutionizing how we model, predict, and control complex systems such as climate, finance, traffic, robotics and autonomy. The most pressing scientific and engineering tasks of the present era are not dependent on empirical models or derivations based on first principles. Increasingly, researchers are turning to data-driven approaches for characterizing and modeling a diverse range of complex systems with the goal of sensing, prediction, estimation, and control.

One such challenge is analyzing the performance of electric machines in a relatively short period of time. An electromagnetic model is needed to perform the design and analysis of an electric machine. This model will be used to evaluate relevant electromagnetic performance characteristics associated with an electric machine such as electromagnetic force, torque, fields, and losses distribution maps. Further, a model capable of providing an accurate and fast performance evaluation is the backbone of an optimization task for achieving objectives such as minimizing the torque ripple, increasing average torque and efficiency. It also plays a central role in designing the motor drive system.

## The Nature of Models

The very word 'model' implies simplification and idealization. A model is an approximation of reality and hence will not reflect all of reality. In 1976, a renowned British statistician named George Box wrote the famous line:

> "All models are wrong, some are useful"

The intuition behind this sentence is that every model is wrong because it is a simplification of reality. However, simplifications of reality can be quite useful. They can help us explain, predict and understand the universe and all its various components. It means useful insights can be provided from models which are not a perfect representation of the phenomena they model. The practical question is how wrong do they have to be in order to be not useful.

### Desirable Model Features

As such, the desirable features of a model can be summarized as follows:

1. **Consistency**: A model should be consistent in its ability to explain past observations and predict future observations.
2. **Computational Efficiency**: A model should be computationally cheaper than traditional methods available for analysis.
3. **Accuracy**: A model that gives highly accurate results but takes a lot of computational time may not be useful. On the other hand, a model that is not as accurate but can produce estimates very quickly can be beneficial.

## Electromagnetic Modeling Challenges

### First Principle Models

Maxwell's equations form the foundation of classical electromagnetism, classical optics, and electric circuits. We can consider models built on solving physics (such as Maxwell's equations) as the First Principle models. Modeling is the mathematical representation of physical phenomena and simulation is the numerical representation of such models on computing machines. Over the years Computer-Aided Design (CAD) has evolved as the field of "using computers to aid in the creation, modification, analysis, or optimization of a design".

Accurate modeling and analysis of an electromagnetic problem is both a challenging and time-consuming process due to several reasons:

1. **Nonlinear Systems**: The system of differential equations for the electromagnetic analysis of electric devices is nonlinear due to the property of saturation in magnetic materials.
2. **Complex Geometry**: The non-linear dependence of inductances on rotor-to-stator angles is reflected in the non-linear relationship between current and fluxes.
3. **Computational Complexity**: Since the induced voltage and electromagnetic torque are proportional to the state variables, such as flux, current, and speed, it makes it impossible to find an analytic solution to the machine system of differential equations.

## Electromagnetic Modeling Approaches

Having identified the challenges associated with the field of electromagnetic modeling and simulation, we can broadly divide the field into three categories of algorithms:

```{figure} ../_static/figures/EM_model_techniques.png
---
name: fig-modeling-techniques
alt: Different electromagnetic modeling techniques
---
Different electromagnetic modeling techniques.
```

### 1. Analytical Methods

Of the three approaches, analytical algorithms are usually the least demanding in terms of computational need and will provide the fastest results. These methods try to find the closed-form expressions for magnetic fields and losses in a motor. However, they face a major challenge in:

- **Geometric Complexity**: Approximating complex geometries and incorporating the effect of iron saturation in their analysis
- **Physical Effects**: They are incapable of modeling skin and proximity effects in the winding and the end winding inductance

Despite their limited capabilities, analytic models are popular because of the low computation power required for their highly simplified expressions and ease of parameterization. They are employed when fast prototyping is needed and the design is evaluated only on the basis of global performance quantities such as torque and forces.

### 2. Physics-based Models

The physics-based models, on the other hand, are founded on established and detailed physical principles and lead to more accurate simulations. They model the interaction of electromagnetic fields with physical objects and the environment. Two such methods are:

#### Magnetic Equivalent Circuits (MEC)

MEC modeling is a popular method in modeling electric machines. It's a special case since it can be considered as an analytical or numerical technique according to how it is applied:

- **As Analytical**: When combined with other analytical modeling techniques such as Maxwell's equations
- **As Numerical**: When the nonlinearity of the magnetic materials is considered, requiring combination with numerical techniques

MECs offer an alternative possibility based on permeance network models comprising reluctance and mmf sources. The MEC can be considered as a reduced order FE which translates a geometrical description of a magnetic device into an electrical circuit description.

#### Finite Element Analysis (FEA)

The present-day Finite Element (FE) method based software suite allows for the accurate analysis of electric machines in three dimensions with coupled fields. It is the method of choice for today's machine designers, in comparison to analytical models and MECs.

**FEA Process Steps**:
1. **Domain Discretization**: Proper discretization optimizes the number of elements and unknowns for the desired solution accuracy
2. **Selection of Interpolation Functions**: Converting a continuous operator problem to a discrete problem
3. **Formulation of the System of Equations**: Methods like Ritz and Galerkin
4. **Solution of the System of Equations**: Using numerical methods like LU Decomposition, Conjugate Gradient Method

```{figure} ../_static/figures/Proposal.pdf
---
name: fig-fem-process
alt: Computational Analysis in FEM
---
Computational Analysis in FEM
```

### 3. Surrogate/Data-driven Models

Another class of models, surrogate or data-driven models, is not necessarily based on first principles. They approximate the relationship between the input parameters and output characteristics and reduce the repetitive bottom-up approach taken by the first principle models. Machine Learning (ML) and curve-fitting-based techniques are some of the approaches in this category of modeling used to predict the performance of electric machines.

**Advantages**:
- Provide a fast prediction of global performance parameters such as average and ripple torque
- Include experimental or Finite Element (FE) model data to develop the model

**Limitations**:
- Based on current literature, these models are not capable of producing information on local effects within the device such as fields, losses, and stresses
- Knowledge of the flux distribution inside the motor for different operating conditions is essential at the design stage to predict the performance

## Comparison Between MEC and FEA

### MEC Advantages
- **Reduced Model Complexity**: Compared to FE
- **Ease of Parameterization**: Simple to adjust parameters
- **Fast Computational Time**: Quick analysis
- **3D Extension**: Extension to 3-D does not expand numerical complexity as quickly as FEA
- **Initial Sizing**: Useful for performance analysis and initial device sizing

### FEA Advantages
- **Fine Resolution**: Offers finer modeling and simulation resolution
- **Complex Analysis**: Enables study of complex issues such as internal faults or localized magnetic saturation
- **Flexible Design**: More flexible tool to study designs incorporating new shapes
- **Accurate**: Generally more accurate than MEC analysis
- **Deployment Time**: Generally requires less deployment time for simulation

### Performance Comparison

| Reference | Method | Force/Torque Comparison |
|-----------|--------|-------------------------|
| Liu et al. (2006) | FEA-MEC | 6% on force between MEC and test, 10% on force between FEA and test |
| Hur et al. (1997) | FEA-MEC | 8.6% on force between 2D MEC and test, 9% on force between 2D FEA and test, 1% on force between 3D MEC and test |
| Leplat et al. (1996) | FEA-MEC | 16.9% on torque between MEC and test, 11.1% on torque between FEA and test |
| Kim et al. (2004) | FEA-MEC | 3-5% on force between FEA and test, 5-35% on force between MEC and test |

## Need for Improved Surrogate Models

Having discussed the advantages and limitations of the popular numerical methods associated with the analysis of electric machines, it is clear that a computationally cheap and accurate field solution is necessary for the analysis of electromagnetic devices such as actuators, transformers, and electrical machines.

**Current Challenges**:
1. **Optimization Overhead**: Usually, for generating an optimal design, we need to perform a few thousand analyses as part of optimization
2. **Computational Expense**: Using FEA methods for modeling and simulation purposes remains a computationally expensive process
3. **Search Space Limitation**: Using FEA directly for design steps dramatically reduces the size of search space and is computationally very expensive

**Deep Learning Opportunity**:
Recent rapid developments in the field of machine learning and big data have opened new venues for pattern recognition and curve fitting in complex problems such as computer vision, dense regression, text analysis, and forecasting. The possibility of applying deep networks to magnetic field estimation could accelerate the solution of such problems.

This chapter will explore the field of Deep Learning (a sub-field of ML) to predict the field distribution for different electromagnetic problems using a bitmap approach.