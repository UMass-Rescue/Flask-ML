import React from 'react';
import clsx from 'clsx';
import ThemedImage from '@theme/ThemedImage';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Easy to Use',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        <div style={{ fontSize: '1.1em', textAlign: 'justify' }}>
        Welcome to Flask-ML! Flask-ML is a Flask extension for running machine learning code
        </div>
      </>
    ),
  },
];

function Feature({ Svg, title, description }) {
  return (
    <div className="row">
      <div className={clsx('col col--4 text--center')}>
        <ThemedImage
          alt="UMass Rescue Lab"
          sources={{
            light: useBaseUrl('img/Rescue+Lab+LogoOL.jpg'),
            dark: useBaseUrl('img/Rescue+Lab+LogoOL.jpg'),
          }}
        />
      </div>
      <div className={clsx('col col--8')}>
        <div className="">
          <p>{description}</p>
        </div>
      </div>
    </div>
  );
}

function FeatureOri({ Svg, title, description }) {
  return (
    <div className={clsx('col col--12')}>
      {/* <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div> */}
      <div className="text--center padding-horiz--md">
        {/* <h3>{title}</h3> */}
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}